// RF-DETR C++ ONNX 目标检测与实例分割推理实现：
// 模型输入：image [1, 3, H, W]（RGB 格式、ImageNet 均值方差归一化）
// 模型输出：dets [1, Q, 4], labels [1, Q, C+1], 可选 masks [1, Q, MH, MW]

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
  std::string output = "results/cpp_rfdetr";                                // 输出结果目录路径
  std::string device = "cpu";                                               // 推理设备（"cpu" 或 "cuda"）
  std::vector<std::string> classes;                                         // 类别名称列表（支持任意类别数量）
  float score_threshold = 0.40F;                                            // 全局置信度阈值
  float iou_threshold = 0.50F;                                              // 全局 IoU 阈值
  float mask_threshold = 0.50F;                                             // 掩码二值化阈值
  std::vector<float> class_conf;                                            // 各类别专属置信度阈值
  std::vector<float> class_iou;                                             // 各类别专属 IoU 阈值
  bool draw = true;                                                         // 是否生成可视化图像
};

// 收集以非 '-' 开头的参数列表项，兼容逗号分隔格式
static std::vector<std::string> collect_tokens(int& i, int argc, char** argv) {
  std::vector<std::string> tokens;
  while (i + 1 < argc && argv[i + 1][0] != '-') {
    tokens.push_back(argv[++i]);
  }
  if (tokens.size() == 1 && tokens.front().find(',') != std::string::npos) {
    std::stringstream ss(tokens.front());
    std::string token;
    tokens.clear();
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) tokens.push_back(token);
    }
  }
  return tokens;
}

// 解析按类别指定的阈值配置参数（例如 CLASS=VALUE）
static std::vector<float> parse_threshold_tokens(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& classes,
    float fallback,
    const std::string& label) {
  std::vector<float> result(classes.size(), fallback);
  std::vector<bool> seen(classes.size(), false);
  for (const auto& token : tokens) {
    auto equal = token.find('=');
    if (equal == std::string::npos) {
      throw std::runtime_error(label + " expects CLASS=VALUE");
    }
    std::string name = token.substr(0, equal);
    auto it = std::find(classes.begin(), classes.end(), name);
    if (it == classes.end()) {
      throw std::runtime_error(label + " unknown class: " + name);
    }
    size_t index = static_cast<size_t>(it - classes.begin());
    if (seen[index]) {
      throw std::runtime_error(label + " duplicate class: " + name);
    }
    float parsed = std::stof(token.substr(equal + 1));
    if (!std::isfinite(parsed) || parsed < 0.0F || parsed > 1.0F ||
        (label == "--class-iou" && parsed <= 0.0F)) {
      throw std::runtime_error(label + " value out of range (0, 1]: " + token);
    }
    result[index] = parsed;
    seen[index] = true;
  }
  return result;
}

// 解析命令行输入参数
static Args parse_args(int argc, char** argv) {
  Args args;
  std::vector<std::string> conf_tokens, iou_tokens, class_tokens;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      std::cout << "Usage: rfdetr_infer --model MODEL.onnx --input IMAGE_OR_DIR --output DIR "
                << "[--device cpu|cuda] [--classes class1,class2,...|classes.names] "
                << "[--score-threshold 0.40] [--iou-threshold 0.50] "
                << "[--class-conf class1=0.8,class2=0.5] [--class-iou class1=0.5,class2=0.5] "
                << "[--draw | --no-draw]\n";
      std::exit(0);
    }
    if (key == "--model" && i + 1 < argc) {
      args.model = argv[++i];
    } else if (key == "--input" && i + 1 < argc) {
      args.input = argv[++i];
    } else if (key == "--output" && i + 1 < argc) {
      args.output = argv[++i];
    } else if (key == "--device" && i + 1 < argc) {
      args.device = argv[++i];
    } else if (key == "--classes") {
      class_tokens = collect_tokens(i, argc, argv);
    } else if (key == "--score-threshold" && i + 1 < argc) {
      args.score_threshold = std::stof(argv[++i]);
    } else if (key == "--iou-threshold" && i + 1 < argc) {
      args.iou_threshold = std::stof(argv[++i]);
    } else if (key == "--mask-threshold" && i + 1 < argc) {
      args.mask_threshold = std::stof(argv[++i]);
    } else if (key == "--class-conf") {
      auto tk = collect_tokens(i, argc, argv);
      conf_tokens.insert(conf_tokens.end(), tk.begin(), tk.end());
    } else if (key == "--class-iou") {
      auto tk = collect_tokens(i, argc, argv);
      iou_tokens.insert(iou_tokens.end(), tk.begin(), tk.end());
    } else if (key == "--draw") {
      args.draw = true;
    } else if (key == "--no-draw") {
      args.draw = false;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  if (args.model.empty() || args.input.empty()) {
    throw std::runtime_error(
        "Usage: rfdetr_infer --model MODEL.onnx --input IMAGE_OR_DIR [--output DIR] [--classes a,b] [--device cpu|cuda]");
  }
  if (args.device != "cpu" && args.device != "cuda") {
    throw std::runtime_error("--device must be cpu or cuda");
  }
  if (!class_tokens.empty()) {
    if (class_tokens.size() == 1 && fs::is_regular_file(class_tokens[0])) {
      std::ifstream file(class_tokens[0]);
      std::string line;
      while (std::getline(file, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const auto first = line.find_first_not_of(" \t\r\n");
        const auto last = line.find_last_not_of(" \t\r\n");
        if (first != std::string::npos) args.classes.push_back(line.substr(first, last - first + 1));
      }
    } else {
      args.classes = class_tokens;
    }
  } else if (fs::is_regular_file("classes.names")) {
    std::ifstream file("classes.names");
    std::string line;
    while (std::getline(file, line)) {
      const size_t comment = line.find('#');
      if (comment != std::string::npos) line.erase(comment);
      const auto first = line.find_first_not_of(" \t\r\n");
      const auto last = line.find_last_not_of(" \t\r\n");
      if (first != std::string::npos) args.classes.push_back(line.substr(first, last - first + 1));
    }
  }

  if (!args.classes.empty()) {
    args.class_conf = parse_threshold_tokens(conf_tokens, args.classes, args.score_threshold, "--class-conf");
    args.class_iou = parse_threshold_tokens(iou_tokens, args.classes, args.iou_threshold, "--class-iou");
  }

  return args;
}

// 转义字符串中的特殊字符以满足 JSON 格式要求
static std::string escape(const std::string& text) {
  std::string result;
  for (char c : text) {
    if (c == '\\' || c == '"') result += '\\';
    result += c;
  }
  return result;
}

// 遍历并收集输入路径下的所有有效图像文件
static std::vector<fs::path> images(const std::string& source) {
  fs::path path(source);
  std::vector<fs::path> result;
  if (fs::is_regular_file(path)) {
    result.push_back(path);
  } else if (fs::is_directory(path)) {
    for (const auto& item : fs::directory_iterator(path)) {
      if (!item.is_regular_file()) continue;
      std::string ext = item.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp" || ext == ".tif" || ext == ".tiff") {
        result.push_back(item.path());
      }
    }
  } else {
    throw std::runtime_error("input does not exist: " + source);
  }
  std::sort(result.begin(), result.end());
  if (result.empty()) throw std::runtime_error("no supported images found: " + source);
  return result;
}

// 图像预处理：转换色彩空间为 RGB、双线性缩放、按 ImageNet 均值方差归一化并排布为 CHW 浮点张量
static std::vector<float> preprocess(const cv::Mat& bgr, int height, int width) {
  cv::Mat rgb, resized;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  resized.convertTo(resized, CV_32F, 1.0 / 255.0);
  const float mean[] = {.485F, .456F, .406F}, stdev[] = {.229F, .224F, .225F};
  std::vector<float> data(static_cast<size_t>(3) * height * width);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        data[static_cast<size_t>(c) * height * width + y * width + x] =
            (resized.at<cv::Vec3f>(y, x)[c] - mean[c]) / stdev[c];
      }
    }
  }
  return data;
}

// Sigmoid 激活函数：将 logit 映射为 (0, 1) 的概率值
static float probability(float value) {
  return 1.0F / (1.0F + std::exp(-value));
}

// 掩码解码函数：将低分辨率 logits 插值还原至原图分辨率，并根据阈值二值化
static std::vector<uint8_t> decode_mask(const float* logits, int mask_h, int mask_w, int image_h, int image_w, float threshold) {
  cv::Mat source(mask_h, mask_w, CV_32F, const_cast<float*>(logits)), resized, binary;
  cv::resize(source, resized, cv::Size(image_w, image_h), 0, 0, cv::INTER_LINEAR);
  // RF-DETR 的 Python 后处理器先对 mask logits 进行双线性插值，再以 0（对应 Sigmoid 阈值）进行二值化。
  cv::threshold(resized, binary, std::log(threshold / (1.0F - threshold)), 1, cv::THRESH_BINARY);
  binary.convertTo(binary, CV_8U);
  return std::vector<uint8_t>(binary.datastart, binary.dataend);
}

// 计算两个矩形边界框的交并比（IoU）
static float box_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float intersection = (a & b).area();
  const float total = a.area() + b.area() - intersection;
  return total <= 0.0F ? 0.0F : intersection / total;
}

// 按类别执行非极大值抑制（Class-wise NMS）
static std::vector<Detection> class_nms(std::vector<Detection> detections, const std::vector<float>& conf, const std::vector<float>& ious) {
  std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> selected;
  for (const auto& candidate : detections) {
    if (candidate.score < conf[candidate.class_id]) continue;
    bool suppressed = false;
    for (const auto& kept : selected) {
      if (kept.class_id == candidate.class_id && ious[candidate.class_id] < 1.0F &&
          box_iou(kept.box, candidate.box) > ious[candidate.class_id]) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) selected.push_back(candidate);
  }
  return selected;
}

// 根据类别 ID 与类别总数获取可视化渲染对应的颜色（基于 HSV 转 RGB）
static cv::Scalar color(int id, size_t num_classes = 1) {
  return get_class_color(id, num_classes);
}

// 渲染检测与分割结果：半透明掩码、掩码轮廓、边界框以及类别置信度标签
static void draw(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classes) {
  cv::Mat overlay = image.clone();
  for (const auto& d : detections) {
    const cv::Scalar c = color(d.class_id, classes.size());
    if (!d.mask.empty()) {
      cv::Mat mask(image.rows, image.cols, CV_8U, const_cast<uint8_t*>(d.mask.data()));
      overlay.setTo(c, mask);
    }
  }
  cv::addWeighted(overlay, 0.40, image, 0.60, 0, image);

  for (const auto& d : detections) {
    const cv::Scalar c = color(d.class_id, classes.size());
    if (!d.mask.empty()) {
      cv::Mat mask(image.rows, image.cols, CV_8U, const_cast<uint8_t*>(d.mask.data()));
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(image, contours, -1, c, 2, cv::LINE_AA);
    }
    cv::rectangle(image, d.box, c, 2, cv::LINE_AA);

    std::string cname = (d.class_id >= 0 && d.class_id < static_cast<int>(classes.size()))
                            ? classes[d.class_id]
                            : ("class_" + std::to_string(d.class_id));
    std::ostringstream ss;
    ss << cname << " " << std::fixed << std::setprecision(2) << d.score;
    std::string text = ss.str();

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int tx = std::max(0, static_cast<int>(d.box.x));
    int ty = std::max(text_size.height + 4, static_cast<int>(d.box.y) - 4);
    cv::rectangle(image, cv::Point(tx, ty - text_size.height - 4),
                  cv::Point(tx + text_size.width + 6, ty + baseline - 2), c, cv::FILLED);
    cv::putText(image, text, cv::Point(tx + 3, ty - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

// 导出推理结果至结构化 JSON 文件（包含目标类别、置信度、边界框、掩码面积、FNV-1a 哈希及 RLE 编码）
static void write_json(const fs::path& path, const fs::path& image, const std::vector<Detection>& detections, const std::vector<std::string>& classes) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("unable to write: " + path.string());
  out << std::setprecision(9) << "{\n  \"image\": \"" << escape(fs::absolute(image).string()) << "\",\n  \"detections\": [\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    uint64_t hash = 1469598103934665603ULL;
    size_t area = 0;
    for (uint8_t v : d.mask) {
      area += v != 0;
      hash ^= v;
      hash *= 1099511628211ULL;
    }
    out << "    {\"class_id\": " << d.class_id << ", \"class_name\": \"" << escape(classes[d.class_id])
        << "\", \"score\": " << d.score << ", \"bbox_xyxy\": [" << d.box.x << ", " << d.box.y << ", "
        << d.box.x + d.box.width << ", " << d.box.y + d.box.height << "], \"mask_area\": " << area
        << ", \"mask_fnv1a64\": \"" << std::hex << std::setw(16) << std::setfill('0') << hash << std::dec
        << std::setfill(' ') << "\", \"mask_rle\": [";
    uint8_t current = 0;
    size_t run = 0;
    bool first = true;
    for (uint8_t v : d.mask) {
      if (v == current) {
        ++run;
      } else {
        if (!first) out << ", ";
        out << run;
        first = false;
        current = v;
        run = 1;
      }
    }
    if (!first) out << ", ";
    out << run << "]}" << (i + 1 == detections.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

// 依据名称查找 ONNX 输出张量节点的索引
static int find_output(const std::vector<std::string>& names, const std::string& expected, int fallback) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == expected) return static_cast<int>(i);
  }
  return fallback;
}

// 主函数：模型加载、输出节点绑定、图像遍历推理及结果导出
int main(int argc, char** argv) {
  try {
    RunLogger logger("RFDETR");
    const Args args = parse_args(argc, argv);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rfdetr");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (args.device == "cuda") {
#ifdef ENABLE_CUDA_PROVIDER
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      options.AppendExecutionProvider_CUDA(cuda_options);
#else
      throw std::runtime_error("this executable was built without CUDA provider support");
#endif
    }

    Ort::Session session(env, args.model.c_str(), options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() < 2) {
      throw std::runtime_error("RF-DETR ONNX requires one input and at least dets/labels outputs");
    }

    const auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 || input_shape[2] <= 0 || input_shape[3] <= 0) {
      throw std::runtime_error("unsupported input shape; expected fixed [1,3,H,W]");
    }
    const int input_h = static_cast<int>(input_shape[2]), input_w = static_cast<int>(input_shape[3]);
    if (input_h % 4 || input_w % 4) {
      throw std::runtime_error("RF-DETR input H/W must be divisible by 4 for mask decoding");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<Ort::AllocatedStringPtr> allocated;
    std::vector<std::string> output_names_str;
    std::vector<const char*> output_names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      allocated.push_back(session.GetOutputNameAllocated(i, allocator));
      output_names_str.emplace_back(allocated.back().get());
      output_names.push_back(allocated.back().get());
    }
    const int det_index = find_output(output_names_str, "dets", 0);
    const int label_index = find_output(output_names_str, "labels", 1);
    const int mask_index = session.GetOutputCount() > 2 ? find_output(output_names_str, "masks", 2) : -1;

    auto image_list = images(args.input);
    fs::create_directories(args.output);
    std::cout << "Running RF-DETR inference on " << image_list.size() << " images from " << args.input << "...\n";

    for (size_t idx = 0; idx < image_list.size(); ++idx) {
      const auto& image_path = image_list[idx];
      cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (image.empty()) {
        std::cerr << "skip unreadable " << image_path << "\n";
        continue;
      }
      std::vector<float> input = preprocess(image, input_h, input_w);
      const std::array<int64_t, 4> dims{{1, 3, input_h, input_w}};
      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), dims.data(), dims.size());
      const char* input_names[] = {input_name.get()};
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names.data(), output_names.size());

      const auto det_shape = outputs[det_index].GetTensorTypeAndShapeInfo().GetShape();
      const auto label_shape = outputs[label_index].GetTensorTypeAndShapeInfo().GetShape();
      if (det_shape.size() != 3 || det_shape[0] != 1 || det_shape[2] != 4 ||
          label_shape.size() != 3 || label_shape[0] != 1 || label_shape[1] != det_shape[1]) {
        throw std::runtime_error("RF-DETR expected dets [1,Q,4] and labels [1,Q,C+1]");
      }
      const int model_classes = static_cast<int>(label_shape[2] - 1);
      std::vector<std::string> eff_classes = args.classes;
      std::vector<float> eff_class_conf = args.class_conf;
      std::vector<float> eff_class_iou = args.class_iou;
      if (eff_classes.empty()) {
        for (int c = 0; c < model_classes; ++c) eff_classes.push_back("class_" + std::to_string(c));
        eff_class_conf.assign(eff_classes.size(), args.score_threshold);
        eff_class_iou.assign(eff_classes.size(), args.iou_threshold);
      } else if (label_shape[2] < static_cast<int64_t>(eff_classes.size() + 1)) {
        throw std::runtime_error("RF-DETR model has " + std::to_string(model_classes) + " classes, but " +
                                 std::to_string(eff_classes.size()) + " classes were configured");
      }
      const int queries = static_cast<int>(det_shape[1]), classes = static_cast<int>(eff_classes.size());
      const float* boxes = outputs[det_index].GetTensorData<float>();
      const float* labels = outputs[label_index].GetTensorData<float>();
      const float* masks = nullptr;
      int mask_h = 0, mask_w = 0;
      if (mask_index >= 0) {
        const auto shape = outputs[mask_index].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != queries || shape[2] <= 0 || shape[3] <= 0) {
          throw std::runtime_error("RF-DETR masks must have shape [1,Q,MH,MW]");
        }
        masks = outputs[mask_index].GetTensorData<float>();
        mask_h = static_cast<int>(shape[2]);
        mask_w = static_cast<int>(shape[3]);
      }

      struct Candidate { int query; int cls; float score; };
      std::vector<Candidate> candidates;
      const int logit_classes = static_cast<int>(label_shape[2]);
      for (int q = 0; q < queries; ++q) {
        for (int c = 0; c < logit_classes; ++c) {
          const float score = probability(labels[static_cast<size_t>(q) * label_shape[2] + c]);
          candidates.push_back({q, c, score});
        }
      }
      std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
      });
      if (candidates.size() > 300) candidates.resize(300);

      std::vector<Detection> detections;
      for (const auto& candidate : candidates) {
        const int q = candidate.query, cls = candidate.cls;
        // RF-DETR 的最后一个 logit 为无目标（no-object）占位符。Python API 在所有
        // logit 中统一选取 top-k，随后在类别名称映射时丢弃该占位符。
        if (cls >= classes) continue;
        const float best = candidate.score;
        if (best <= eff_class_conf[cls]) continue;

        const float cx = boxes[q * 4] * image.cols, cy = boxes[q * 4 + 1] * image.rows;
        const float bw = boxes[q * 4 + 2] * image.cols, bh = boxes[q * 4 + 3] * image.rows;
        Detection d;
        d.class_id = cls;
        d.score = best;
        d.box = cv::Rect2f(
            std::clamp(cx - bw / 2, 0.0F, static_cast<float>(image.cols - 1)),
            std::clamp(cy - bh / 2, 0.0F, static_cast<float>(image.rows - 1)),
            std::max(0.0F, std::min(cx + bw / 2, static_cast<float>(image.cols)) - std::max(cx - bw / 2, 0.0F)),
            std::max(0.0F, std::min(cy + bh / 2, static_cast<float>(image.rows)) - std::max(cy - bh / 2, 0.0F)));
        if (masks) {
          d.mask = decode_mask(masks + static_cast<size_t>(q) * mask_h * mask_w, mask_h, mask_w, image.rows, image.cols, args.mask_threshold);
        } else {
          d.mask.assign(static_cast<size_t>(image.rows) * image.cols, 0);
        }
        detections.push_back(std::move(d));
      }

      detections = class_nms(std::move(detections), eff_class_conf, eff_class_iou);

      const fs::path stem = fs::path(args.output) / image_path.stem();
      write_json(stem.string() + ".json", image_path, detections, eff_classes);

      if (args.draw) {
        cv::Mat visualization = image.clone();
        draw(visualization, detections, eff_classes);
        const fs::path vis_path = fs::path(args.output) / (image_path.stem().string() + "_vis.jpg");
        cv::imwrite(vis_path.string(), visualization, {cv::IMWRITE_JPEG_QUALITY, 95});
        std::cout << "[" << (idx + 1) << "/" << image_list.size() << "] " << image_path.filename().string()
                  << ": " << detections.size() << " detections -> saved " << vis_path.filename().string() << "\n";
      } else {
        std::cout << "[" << (idx + 1) << "/" << image_list.size() << "] " << image_path.filename().string()
                  << ": " << detections.size() << " detections\n";
      }
    }
    std::cout << "\nInference completed successfully. All outputs saved to: " << args.output << "\n";
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << "ONNX Runtime error: " << error.what() << "\n";
    return 2;
  } catch (const cv::Exception& error) {
    std::cerr << "OpenCV error: " << error.what() << "\n";
    return 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
