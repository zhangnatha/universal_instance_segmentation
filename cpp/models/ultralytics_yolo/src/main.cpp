// Ultralytics YOLO-seg (YOLOv8-seg / YOLO11-seg) C++ ONNX 实例分割推理实现：
// 模型输入：image [1, 3, H, W]（采用 Letterbox 等比例缩放填充，RGB 归一化至 [0, 1]）
// 模型输出：Rank-3 预测分支（包含坐标、类别概率与掩码系数）及 Rank-4 原型掩码分支（Prototypes）

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../../common/run_logger.hpp"
#include "../../../common/colors.hpp"

namespace fs = std::filesystem;

// 单个目标的检测与分割预测结果
struct Detection {
  int class_id = 0;              // 类别 ID
  float score = 0.0F;            // 置信度分数
  cv::Rect2f box;                // 目标边界框（浮点矩形）
  std::vector<uint8_t> mask;     // 二值分割掩码（尺寸与原图一致，单通道）
};

// 命令行参数结构体
struct Args {
  std::string model;                                                        // ONNX 模型文件路径
  std::string input;                                                        // 输入图像或目录路径
  std::string output = "results/cpp_yolo";                                  // 输出结果目录路径
  std::string device = "cpu";                                               // 推理设备（"cpu" 或 "cuda"）
  std::vector<std::string> classes;                                         // 类别名称列表（支持任意类别数量）
  float score_threshold = 0.25F;                                            // 全局置信度阈值
  float iou_threshold = 0.45F;                                              // 全局 IoU 阈值
  float mask_threshold = 0.5F;                                              // 掩码二值化阈值
  std::vector<float> class_conf, class_iou;                                 // 各类别专属置信度与 IoU 阈值
  bool draw = true;                                                         // 是否生成可视化图像
};

// 解析以逗号分隔或文件定义的类别名称字符串列表（支持任意类别数量）
static std::vector<std::string> split_classes(const std::string& value) {
  if (fs::is_regular_file(value)) {
    std::ifstream file(value);
    std::vector<std::string> result;
    std::string line;
    while (std::getline(file, line)) {
      const size_t comment = line.find('#');
      if (comment != std::string::npos) line.erase(comment);
      const auto first = line.find_first_not_of(" \t\r\n");
      const auto last = line.find_last_not_of(" \t\r\n");
      if (first != std::string::npos) result.push_back(line.substr(first, last - first + 1));
    }
    if (!result.empty()) return result;
  }
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const auto first = item.find_first_not_of(" \t");
    const auto last = item.find_last_not_of(" \t");
    if (first != std::string::npos) result.push_back(item.substr(first, last - first + 1));
  }
  if (result.empty()) throw std::runtime_error("--classes must contain at least one name");
  return result;
}

// 解析按类别指定的阈值参数（例如 CLASS=VALUE[,CLASS=VALUE]）
static std::vector<float> parse_class_thresholds(const std::string& value, const std::vector<std::string>& classes, float fallback, const std::string& label) {
  std::vector<float> result(classes.size(), fallback); std::vector<bool> seen(classes.size(), false); std::stringstream stream(value); std::string item;
  while (std::getline(stream, item, ',')) { const auto equal = item.find('='); if (equal == std::string::npos) throw std::runtime_error(label + " expects CLASS=VALUE[,CLASS=VALUE]"); const std::string name = item.substr(0, equal); auto it = std::find(classes.begin(), classes.end(), name); if (it == classes.end()) throw std::runtime_error(label + " contains unknown class: " + name); const size_t index = static_cast<size_t>(it - classes.begin()); if (seen[index]) throw std::runtime_error(label + " contains duplicate class: " + name); float parsed = std::stof(item.substr(equal + 1)); if (!std::isfinite(parsed) || parsed < 0.0F || parsed > 1.0F || (label == "--class-iou" && parsed <= 0.0F)) throw std::runtime_error(label + " value must be in (0,1]"); result[index] = parsed; seen[index] = true; }
  return result;
}

// 解析命令行参数并进行合法性校验
static Args parse_args(int argc, char** argv) {
  Args args;
  std::string conf_str, iou_str;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      std::cout << "Usage: ultralytics_yolo_infer --model MODEL.onnx --input IMAGE_OR_DIR --output DIR "
                << "[--device cpu|cuda] [--classes class1,class2,...|classes.names] "
                << "[--score-threshold 0.25] [--iou-threshold 0.5] "
                << "[--class-conf class1=0.8,class2=0.5] [--class-iou class1=0.5,class2=0.5] "
                << "[--draw | --no-draw]\n";
      std::exit(0);
    }
    if ((key == "--model" || key == "--input" || key == "--output" || key == "--device" ||
         key == "--classes" || key == "--score-threshold" || key == "--iou-threshold" || key == "--class-conf" || key == "--class-iou" ||
         key == "--mask-threshold") && i + 1 >= argc)
      throw std::runtime_error("Missing value for argument: " + key);
    if (key == "--model") args.model = argv[++i];
    else if (key == "--input") args.input = argv[++i];
    else if (key == "--output") args.output = argv[++i];
    else if (key == "--device") args.device = argv[++i];
    else if (key == "--classes") args.classes = split_classes(argv[++i]);
    else if (key == "--score-threshold") args.score_threshold = std::stof(argv[++i]);
    else if (key == "--iou-threshold") args.iou_threshold = std::stof(argv[++i]);
    else if (key == "--class-conf") conf_str = argv[++i];
    else if (key == "--class-iou") iou_str = argv[++i];
    else if (key == "--mask-threshold") args.mask_threshold = std::stof(argv[++i]);
    else if (key == "--draw") args.draw = true;
    else if (key == "--no-draw") args.draw = false;
    else throw std::runtime_error("Unknown argument: " + key);
  }
  if (args.model.empty() || args.input.empty())
    throw std::runtime_error("Usage: ultralytics_yolo_infer --model MODEL.onnx --input IMAGE_OR_DIR "
                             "[--output DIR] [--classes a,b] [--device cpu|cuda]");
  if (args.device != "cpu" && args.device != "cuda") throw std::runtime_error("--device must be cpu or cuda");
  if (args.score_threshold < 0 || args.score_threshold > 1 || args.iou_threshold < 0 || args.iou_threshold > 1 ||
      args.mask_threshold < 0 || args.mask_threshold > 1)
    throw std::runtime_error("thresholds must be between 0 and 1");

  if (args.classes.empty() && fs::is_regular_file("classes.names")) {
    args.classes = split_classes("classes.names");
  }
  if (!args.classes.empty()) {
    if (!conf_str.empty()) args.class_conf = parse_class_thresholds(conf_str, args.classes, args.score_threshold, "--class-conf");
    else args.class_conf.assign(args.classes.size(), args.score_threshold);
    if (!iou_str.empty()) args.class_iou = parse_class_thresholds(iou_str, args.classes, args.iou_threshold, "--class-iou");
    else args.class_iou.assign(args.classes.size(), args.iou_threshold);
  }
  return args;
}

// 转义字符串中的特殊字符用于构造合法 JSON
static std::string json_escape(const std::string& value) {
  std::string result;
  for (const char c : value) {
    if (c == '\\' || c == '"') result += '\\';
    result += c;
  }
  return result;
}

// 遍历收集输入路径下的所有图像文件列表
static std::vector<fs::path> image_paths(const std::string& source) {
  const fs::path path(source);
  std::vector<fs::path> files;
  if (fs::is_regular_file(path)) files.push_back(path);
  else if (fs::is_directory(path)) {
    for (const auto& item : fs::directory_iterator(path)) {
      if (!item.is_regular_file()) continue;
      std::string ext = item.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp" || ext == ".tif" || ext == ".tiff")
        files.push_back(item.path());
    }
  } else throw std::runtime_error("input does not exist: " + source);
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error("no supported images found in: " + source);
  return files;
}

// Letterbox 缩放预处理结果结构体
struct Letterbox {
  cv::Mat image;                 // 填充后的 RGB 浮点张量图像
  float scale = 1.0F;            // 缩放比例
  int pad_x = 0;                 // 水平方向填充像素偏移
  int pad_y = 0;                 // 垂直方向填充像素偏移
};

// 执行 Letterbox 等比例缩放，并以灰度值 114 进行四周居中填充
static Letterbox preprocess(const cv::Mat& bgr, int width, int height) {
  if (bgr.empty() || bgr.channels() != 3) throw std::runtime_error("input must be a 3-channel color image");
  const float scale = std::min(static_cast<float>(width) / bgr.cols, static_cast<float>(height) / bgr.rows);
  const int resized_w = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
  const int resized_h = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
  cv::Mat resized, canvas(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(bgr, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
  const int pad_x = (width - resized_w) / 2;
  const int pad_y = (height - resized_h) / 2;
  resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_w, resized_h)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32F, 1.0 / 255.0);
  return {canvas, scale, pad_x, pad_y};
}

// 计算置信度数值（若不在 [0, 1] 范围则应用 Sigmoid 激活函数）
static float score_value(float value) {
  return value >= 0.0F && value <= 1.0F ? value : 1.0F / (1.0F + std::exp(-value));
}

// 计算两个边界框的交并比（IoU）
static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float intersection = (a & b).area();
  const float union_area = a.area() + b.area() - intersection;
  return union_area <= 0.0F ? 0.0F : intersection / union_area;
}

// 按类别执行非极大值抑制（Class-wise NMS）
static std::vector<Detection> class_nms(std::vector<Detection> candidates, const std::vector<float>& class_conf, const std::vector<float>& class_iou) {
  std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> selected;
  for (const Detection& candidate : candidates) {
    bool suppressed = false;
    for (const Detection& kept : selected)
      if (candidate.class_id == kept.class_id && class_iou[candidate.class_id] < 1.0F && iou(candidate.box, kept.box) > class_iou[candidate.class_id]) { suppressed = true; break; }
    if (!suppressed && candidate.score >= class_conf[candidate.class_id]) selected.push_back(candidate);
  }
  return selected;
}

// 根据预测掩码系数与原型掩码（Proto-mask）矩阵相乘，还原原图尺寸二值分割掩码
static std::vector<uint8_t> make_mask(const float* coefficients, int coefficient_count,
                                      const float* proto, int proto_h, int proto_w,
                                      const Detection& detection, const Letterbox& letterbox,
                                      int image_h, int image_w, float threshold) {
  cv::Mat logits(proto_h, proto_w, CV_32F, cv::Scalar(0));
  for (int c = 0; c < coefficient_count; ++c) {
    cv::Mat channel(proto_h, proto_w, CV_32F, const_cast<float*>(proto + static_cast<size_t>(c) * proto_h * proto_w));
    logits += coefficients[c] * channel;
  }
  cv::Mat probabilities, canvas_mask;
  cv::exp(-logits, probabilities);
  probabilities = 1.0F / (1.0F + probabilities);
  cv::resize(probabilities, canvas_mask, cv::Size(letterbox.image.cols, letterbox.image.rows), 0, 0, cv::INTER_LINEAR);
  cv::Mat mask(image_h, image_w, CV_8U, cv::Scalar(0));
  for (int y = 0; y < image_h; ++y) {
    for (int x = 0; x < image_w; ++x) {
      const int canvas_x = std::clamp(static_cast<int>(std::round(x * letterbox.scale + letterbox.pad_x)), 0, canvas_mask.cols - 1);
      const int canvas_y = std::clamp(static_cast<int>(std::round(y * letterbox.scale + letterbox.pad_y)), 0, canvas_mask.rows - 1);
      const bool inside = x >= detection.box.x && x <= detection.box.x + detection.box.width &&
                          y >= detection.box.y && y <= detection.box.y + detection.box.height;
      mask.at<uint8_t>(y, x) = inside && canvas_mask.at<float>(canvas_y, canvas_x) >= threshold ? 1 : 0;
    }
  }
  return std::vector<uint8_t>(mask.datastart, mask.dataend);
}

// 根据类别 ID 与类别总数获取可视化渲染对应的颜色（基于 HSV 转 RGB）
static cv::Scalar color_for(int class_id, size_t num_classes = 1) {
  return get_class_color(class_id, num_classes);
}

// 导出推理结果至结构化 JSON 文件（包含检测框、掩码面积、FNV-1a 哈希及 RLE 编码）
static void write_json(const fs::path& path, const fs::path& image_path,
                       const std::vector<Detection>& detections, const std::vector<std::string>& classes) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("unable to write: " + path.string());
  out << std::setprecision(9) << "{\n  \"image\": \"" << json_escape(fs::absolute(image_path).string()) << "\",\n  \"detections\": [\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    uint64_t hash = 1469598103934665603ULL; size_t area = 0;
    for (uint8_t v : d.mask) { area += v != 0; hash ^= v; hash *= 1099511628211ULL; }
    out << "    {\"class_id\": " << d.class_id << ", \"class_name\": \"" << json_escape(classes[d.class_id])
        << "\", \"score\": " << d.score << ", \"bbox_xyxy\": [" << d.box.x << ", " << d.box.y << ", "
        << d.box.x + d.box.width << ", " << d.box.y + d.box.height << "], \"mask_area\": " << area
        << ", \"mask_fnv1a64\": \"" << std::hex << std::setw(16) << std::setfill('0') << hash << std::dec << std::setfill(' ') << "\", \"mask_rle\": [";
    uint8_t current = 0; size_t run = 0; bool first = true;
    for (uint8_t v : d.mask) { if (v == current) ++run; else { if (!first) out << ", "; out << run; first = false; current = v; run = 1; } }
    if (!first) out << ", ";
    out << run << "]}" << (i + 1 == detections.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

// 在图像上绘制半透明掩码覆盖层、边界框及类别置信度标签
static void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classes) {
  for (const auto& d : detections) {
    const cv::Scalar color = color_for(d.class_id, classes.size());
    cv::Mat mask(image.rows, image.cols, CV_8U, const_cast<uint8_t*>(d.mask.data()));
    cv::Mat layer = image.clone(); layer.setTo(color, mask); cv::addWeighted(layer, .28, image, .72, 0, image);
    cv::rectangle(image, d.box, color, 2);
    std::ostringstream label; label << classes[d.class_id] << " " << std::fixed << std::setprecision(2) << d.score;
    cv::putText(image, label.str(), cv::Point(static_cast<int>(d.box.x), std::max(15, static_cast<int>(d.box.y) - 4)), cv::FONT_HERSHEY_SIMPLEX, .5, color, 1, cv::LINE_AA);
  }
}

// 张量元数据包装结构体
struct TensorInfo { std::vector<int64_t> shape; const float* data = nullptr; };

// 获取张量元数据并校验数据类型为 float32
static TensorInfo tensor_info(const Ort::Value& value, const std::string& role) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error(role + " must be float32");
  return {info.GetShape(), value.GetTensorData<float>()};
}

// 主函数：模型加载、动态输出分支匹配、图像推理、Proto-mask 重建及可视化
int main(int argc, char** argv) {
  try {
    RunLogger logger("YOLO");
    const Args args = parse_args(argc, argv);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ultralytics_yolo");
    Ort::SessionOptions options; options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (args.device == "cuda") {
#ifdef ENABLE_CUDA_PROVIDER
      OrtCUDAProviderOptions cuda_options{}; cuda_options.device_id = 0; options.AppendExecutionProvider_CUDA(cuda_options);
#else
      throw std::runtime_error("this executable was built without CUDA provider support");
#endif
    }
    Ort::Session session(env, args.model.c_str(), options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() < 2) throw std::runtime_error("YOLO-seg ONNX requires one input and detection+prototype outputs");
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0)
      throw std::runtime_error("unsupported input shape; expected fixed [1,3,H,W]");
    const int input_h = static_cast<int>(shape[2]), input_w = static_cast<int>(shape[3]);
    std::vector<Ort::AllocatedStringPtr> names;
    std::vector<const char*> output_names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) { names.push_back(session.GetOutputNameAllocated(i, allocator)); output_names.push_back(names.back().get()); }
    for (const auto& image_path : image_paths(args.input)) {
      cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR); if (image.empty()) { std::cerr << "skip unreadable " << image_path << "\n"; continue; }
      const Letterbox letterbox = preprocess(image, input_w, input_h);
      const std::array<int64_t, 4> input_dims{{1, 3, input_h, input_w}};
      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory, reinterpret_cast<float*>(letterbox.image.data), static_cast<size_t>(input_h) * input_w * 3, input_dims.data(), input_dims.size());
      const char* input_names[] = {input_name.get()};
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names.data(), output_names.size());
      int detection_index = -1, prototype_index = -1, mask_channels = 0, proto_h = 0, proto_w = 0;
      TensorInfo detection{}, prototype{};
      for (size_t i = 0; i < outputs.size(); ++i) {
        TensorInfo current = tensor_info(outputs[i], "output");
        if (current.shape.size() == 4 && current.shape[0] == 1 && current.shape[1] > 0 && current.shape[2] > 0 && current.shape[3] > 0) {
          if (prototype_index >= 0) throw std::runtime_error("ambiguous YOLO prototype outputs");
          prototype_index = static_cast<int>(i); prototype = current; mask_channels = static_cast<int>(current.shape[1]); proto_h = static_cast<int>(current.shape[2]); proto_w = static_cast<int>(current.shape[3]);
        } else if (current.shape.size() == 3 && current.shape[0] == 1) {
          if (detection_index >= 0) throw std::runtime_error("ambiguous YOLO detection outputs");
          detection_index = static_cast<int>(i); detection = current;
        }
      }
      if (detection_index < 0 || prototype_index < 0) throw std::runtime_error("YOLO-seg outputs must contain rank-3 detections and rank-4 prototypes");
      const int detection_channels = static_cast<int>(std::max(detection.shape[1], detection.shape[2]));
      const bool channels_first = detection.shape[1] == detection_channels;
      const int anchors = static_cast<int>(channels_first ? detection.shape[2] : detection.shape[1]);
      std::vector<std::string> eff_classes = args.classes;
      std::vector<float> eff_class_conf = args.class_conf;
      std::vector<float> eff_class_iou = args.class_iou;
      if (eff_classes.empty()) {
        const int detected_no_obj = detection_channels - mask_channels - 4;
        const int detected_obj = detection_channels - mask_channels - 5;
        const int detected_classes = (detected_obj > 0 && detected_obj < 2000) ? detected_obj : detected_no_obj;
        if (detected_classes <= 0) throw std::runtime_error("Unable to infer number of classes from YOLO output shape");
        for (int c = 0; c < detected_classes; ++c) eff_classes.push_back("class_" + std::to_string(c));
        eff_class_conf.assign(eff_classes.size(), args.score_threshold);
        eff_class_iou.assign(eff_classes.size(), args.iou_threshold);
      }
      const int class_count = static_cast<int>(eff_classes.size());
      const int expected_no_objectness = 4 + class_count + mask_channels;
      const int expected_objectness = 5 + class_count + mask_channels;
      bool has_objectness = false;
      if (detection_channels == expected_objectness) has_objectness = true;
      else if (detection_channels != expected_no_objectness)
        throw std::runtime_error("YOLO detection channels do not match classes/prototype channels: got " + std::to_string(detection_channels) + ", expected " + std::to_string(expected_no_objectness) + " or " + std::to_string(expected_objectness));
      const auto value_at = [&](int row, int channel) -> float { return channels_first ? detection.data[static_cast<size_t>(channel) * anchors + row] : detection.data[static_cast<size_t>(row) * detection_channels + channel]; };
      std::vector<Detection> candidates;
      for (int row = 0; row < anchors; ++row) {
        float best_score = 0.0F; int best_class = -1;
        const int class_start = has_objectness ? 5 : 4;
        for (int cls = 0; cls < class_count; ++cls) { const float score = score_value(value_at(row, class_start + cls)); if (score > best_score) { best_score = score; best_class = cls; } }
        const float objectness = has_objectness ? score_value(value_at(row, 4)) : 1.0F;
        const float score = best_score * objectness; if (best_class < 0 || score < eff_class_conf[best_class]) continue;
        float cx = value_at(row, 0), cy = value_at(row, 1), bw = value_at(row, 2), bh = value_at(row, 3);
        if (std::max({std::abs(cx), std::abs(cy), std::abs(bw), std::abs(bh)}) <= 2.0F) { cx *= input_w; cy *= input_h; bw *= input_w; bh *= input_h; }
        const float x0 = (cx - bw / 2.0F - letterbox.pad_x) / letterbox.scale;
        const float y0 = (cy - bh / 2.0F - letterbox.pad_y) / letterbox.scale;
        const float x1 = (cx + bw / 2.0F - letterbox.pad_x) / letterbox.scale;
        const float y1 = (cy + bh / 2.0F - letterbox.pad_y) / letterbox.scale;
        Detection d; d.class_id = best_class; d.score = score; d.box = cv::Rect2f(std::clamp(x0, 0.0F, static_cast<float>(image.cols - 1)), std::clamp(y0, 0.0F, static_cast<float>(image.rows - 1)), std::max(0.0F, std::min(x1, static_cast<float>(image.cols)) - std::max(x0, 0.0F)), std::max(0.0F, std::min(y1, static_cast<float>(image.rows)) - std::max(y0, 0.0F)));
        std::vector<float> coeff(static_cast<size_t>(mask_channels)); for (int c = 0; c < mask_channels; ++c) coeff[c] = value_at(row, class_start + class_count + c);
        d.mask = make_mask(coeff.data(), mask_channels, prototype.data + static_cast<size_t>(0), proto_h, proto_w, d, letterbox, image.rows, image.cols, args.mask_threshold);
        candidates.push_back(std::move(d));
      }
      auto detections = class_nms(std::move(candidates), eff_class_conf, eff_class_iou);
      fs::create_directories(args.output);
      const fs::path stem = fs::path(args.output) / image_path.stem();
      write_json(stem.string() + ".json", image_path, detections, eff_classes);
      if (args.draw) { cv::Mat visualization = image.clone(); draw_detections(visualization, detections, eff_classes); cv::imwrite((stem.string() + image_path.extension().string()), visualization); }
      std::cout << "processed " << image_path << ": " << detections.size() << " detections\n";
    }
    return 0;
  } catch (const Ort::Exception& error) { std::cerr << "ONNX Runtime error: " << error.what() << "\n"; return 2; }
    catch (const cv::Exception& error) { std::cerr << "OpenCV error: " << error.what() << "\n"; return 3; }
    catch (const std::exception& error) { std::cerr << "Error: " << error.what() << "\n"; return 1; }
}
