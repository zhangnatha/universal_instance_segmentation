// Detectron2 Mask R-CNN C++ ONNX 推理实现：
// 支持单图推理、批量目录推理以及交互式服务（Server）模式。
// 模型输入：image [3, H, W] (BGR 平面浮点张量)
// 模型输出：boxes [N, 4], scores [N], classes [N], mask_probs [N, 1, MH, MW]

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

// 单个目标的检测与分割预测结果
struct Detection {
  int class_id = 0;                                               // 类别 ID
  float score = 0.0F;                                             // 置信度分数
  std::array<float, 4> box{};                                     // 边界框坐标 [x0, y0, x1, y1]
  std::vector<uint8_t> mask;                                      // 二值分割掩码（尺寸与原图一致，单通道）
  double angle_degrees = std::numeric_limits<double>::quiet_NaN();// 目标主方向角度（度数）
  int center_x = 0;                                               // 目标中心 X 坐标
  int center_y = 0;                                               // 目标中心 Y 坐标
};

namespace fs = std::filesystem;

// 命令行参数结构体
struct Arguments {
  std::string model;                                              // ONNX 模型文件路径
  std::string input;                                              // 输入图像或目录路径
  std::string output;                                             // 输出结果保存路径或目录
  std::string device = "cpu";                                     // 推理设备（"cpu" 或 "cuda"）
  std::vector<std::string> classes;                               // 类别名称列表（支持任意类别数量）
  double score_threshold = 0.5;                                   // 全局置信度阈值
  std::vector<double> class_conf, class_iou;                      // 各类别专属置信度与 IoU 阈值
  bool serve = false;                                             // 是否启用服务模式（从标准输入逐行读取路径）
  bool draw = true;                                               // 是否保存可视化渲染结果图像
};

void create_directories(const std::string& directory);

#include "../../../common/run_logger.hpp"
#include "../../../common/colors.hpp"

// 获取路径中的父级目录（兼容 Linux 与 Windows 路径分隔符）
std::string parent_path(const std::string& path) {
  const std::string::size_type slash = path.find_last_of("/\\");
  return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

// 获取路径中的文件名
std::string filename(const std::string& path) {
  const std::string::size_type slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

// 拼接目录与文件名路径
std::string join_path(const std::string& directory, const std::string& name) {
  if (directory.empty() || directory == ".") return directory.empty() ? name : "./" + name;
  const char tail = directory[directory.size() - 1];
  return directory + (tail == '/' || tail == '\\' ? "" : "/") + name;
}

// 递归创建所需目录（如果不存在，跨平台安全实现）
void create_directories(const std::string& directory) {
  if (directory.empty()) return;
  std::error_code ec;
  fs::create_directories(directory, ec);
  if (ec) {
    throw std::runtime_error("Unable to create directory: " + directory + " (" + ec.message() + ")");
  }
}

// 判断文件扩展名是否为受支持的图像格式
bool is_supported_image(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
         extension == ".bmp" || extension == ".webp" || extension == ".tif" ||
         extension == ".tiff";
}

// 获取指定路径下的所有图像文件（支持单个图像文件或遍历目录）
std::vector<std::string> image_paths(const std::string& source) {
  const fs::path path(source);
  std::vector<std::string> paths;
  if (fs::is_regular_file(path)) {
    if (!is_supported_image(path))
      throw std::runtime_error("Input is not a supported image: " + source +
                               " (expected BMP, PNG, JPEG, TIFF or WebP)");
    paths.push_back(path.string());
  } else if (fs::is_directory(path)) {
    for (const fs::directory_entry& entry : fs::directory_iterator(path)) {
      if (entry.is_regular_file() && is_supported_image(entry.path()))
        paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty())
      throw std::runtime_error("No supported images found in input directory: " + source);
  } else {
    throw std::runtime_error("Input does not exist: " + source);
  }
  return paths;
}

// 保存图像文件，自动创建上级目录
void write_image(const std::string& path, const cv::Mat& image) {
  create_directories(parent_path(path));
  if (!cv::imwrite(path, image))
    throw std::runtime_error("Unable to write image: " + path);
}

// 替换文件扩展名（跨平台兼容）
std::string replace_extension(const std::string& path, const std::string& extension) {
  const std::string::size_type slash = path.find_last_of("/\\");
  const std::string::size_type dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return path + extension;
  return path.substr(0, dot) + extension;
}

// 解析类别列表参数（支持逗号分隔字符串或类别定义文本文件）
std::vector<std::string> split_classes(const std::string& value) {
  if (fs::is_regular_file(value)) {
    std::ifstream file(value);
    std::vector<std::string> result;
    std::string line;
    while (std::getline(file, line)) {
      const size_t comment = line.find('#');
      if (comment != std::string::npos) line.erase(comment);
      const std::string::size_type first = line.find_first_not_of(" \t\r\n");
      const std::string::size_type last = line.find_last_not_of(" \t\r\n");
      if (first != std::string::npos) result.push_back(line.substr(first, last - first + 1));
    }
    if (!result.empty()) return result;
  }
  std::vector<std::string> result;
  std::istringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string::size_type first = item.find_first_not_of(" \t");
    const std::string::size_type last = item.find_last_not_of(" \t");
    if (first != std::string::npos) result.push_back(item.substr(first, last - first + 1));
  }
  if (result.empty()) throw std::runtime_error("--classes must contain at least one name");
  return result;
}

// 解析按类别指定的阈值键值对（例如 CLASS=VALUE[,CLASS=VALUE]）
std::vector<double> parse_class_thresholds(const std::string& value, const std::vector<std::string>& classes, double fallback, const std::string& label) {
  std::vector<double> result(classes.size(), fallback); std::vector<bool> seen(classes.size(), false); std::istringstream stream(value); std::string item;
  while (std::getline(stream, item, ',')) { const auto equal = item.find('='); if (equal == std::string::npos) throw std::runtime_error(label + " expects CLASS=VALUE[,CLASS=VALUE]"); const std::string name = item.substr(0, equal); auto it = std::find(classes.begin(), classes.end(), name); if (it == classes.end()) throw std::runtime_error(label + " contains unknown class: " + name); const size_t index = static_cast<size_t>(it - classes.begin()); if (seen[index]) throw std::runtime_error(label + " contains duplicate class: " + name); const double parsed = std::stod(item.substr(equal + 1)); if (!std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0 || (label == "--class-iou" && parsed <= 0.0)) throw std::runtime_error(label + " value must be in (0,1]"); result[index] = parsed; seen[index] = true; }
  return result;
}

// 计算两个矩形边界框的交并比（IoU）
double bbox_iou(const std::array<float, 4>& left, const std::array<float, 4>& right) {
  const double x0 = std::max<double>(left[0], right[0]);
  const double y0 = std::max<double>(left[1], right[1]);
  const double x1 = std::min<double>(left[2], right[2]);
  const double y1 = std::min<double>(left[3], right[3]);
  const double intersection = std::max(0.0, x1 - x0) * std::max(0.0, y1 - y0);
  const double left_area = std::max(0.0, static_cast<double>(left[2] - left[0])) *
                           std::max(0.0, static_cast<double>(left[3] - left[1]));
  const double right_area = std::max(0.0, static_cast<double>(right[2] - right[0])) *
                            std::max(0.0, static_cast<double>(right[3] - right[1]));
  const double union_area = left_area + right_area - intersection;
  return union_area > 0.0 ? intersection / union_area : 0.0;
}

// 对检测候选结果执行按类别的非极大值抑制（Class-wise NMS）
std::vector<Detection> class_nms(std::vector<Detection> candidates,
                                  const std::vector<double>& class_conf,
                                  const std::vector<double>& class_iou) {
  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& left, const Detection& right) {
              return left.score > right.score;
            });
  std::vector<Detection> selected;
  for (const Detection& candidate : candidates) {
    if (candidate.score < class_conf[static_cast<size_t>(candidate.class_id)]) continue;
    bool suppressed = false;
    for (const Detection& kept : selected) {
      const size_t class_index = static_cast<size_t>(candidate.class_id);
      if (kept.class_id == candidate.class_id && class_iou[class_index] < 1.0 &&
          bbox_iou(kept.box, candidate.box) > class_iou[class_index]) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) selected.push_back(candidate);
  }
  return selected;
}

// 解析命令行参数并进行合法性校验
Arguments parse_arguments(int argc, char** argv) {
  Arguments args;
  std::string conf_str, iou_str;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      std::cout << "Usage: detectron2_maskrcnn_infer --model MODEL --input IMAGE_OR_DIR --output DIR "
                << "[--device cpu|cuda] [--classes class1,class2,...|classes.names] "
                << "[--score-threshold 0.5] [--class-conf class1=0.8,class2=0.5] "
                << "[--class-iou class1=0.5,class2=0.5] [--draw | --no-draw] [--serve]\n";
      std::exit(0);
    }
    if ((key == "--model" || key == "--input" || key == "--output" ||
         key == "--device" || key == "--classes" || key == "--score-threshold" || key == "--class-conf" || key == "--class-iou") && i + 1 >= argc)
      throw std::runtime_error("Missing value for argument: " + key);
    if (key == "--model") args.model = argv[++i];
    else if (key == "--input") args.input = argv[++i];
    else if (key == "--output") args.output = argv[++i];
    else if (key == "--device") args.device = argv[++i];
    else if (key == "--classes") args.classes = split_classes(argv[++i]);
    else if (key == "--score-threshold") args.score_threshold = std::stod(argv[++i]);
    else if (key == "--class-conf") conf_str = argv[++i];
    else if (key == "--class-iou") iou_str = argv[++i];
    else if (key == "--draw") args.draw = true;
    else if (key == "--no-draw") args.draw = false;
    else if (key == "--serve") args.serve = true;
    else throw std::runtime_error("Unknown argument: " + key);
  }
  if (args.model.empty() || args.output.empty() || (!args.serve && args.input.empty()))
    throw std::runtime_error(
        "Usage: detectron2_maskrcnn_infer --model MODEL --input IMAGE --output IMAGE "
        "[--device cpu|cuda] [--classes class1,class2,...|classes.names] "
        "[--score-threshold 0.5] [--class-conf class1=0.5,class2=0.4] "
        "[--class-iou class1=0.5,class2=1.0]\n"
        "Server mode: detectron2_maskrcnn_infer --model MODEL --output DIRECTORY --serve "
        "[--device cpu|cuda]");
  if (args.device != "cpu" && args.device != "cuda")
    throw std::runtime_error("--device must be cpu or cuda");
  if (!(args.score_threshold >= 0.0 && args.score_threshold <= 1.0))
    throw std::runtime_error("--score-threshold must be between 0 and 1");

  if (args.classes.empty() && fs::is_regular_file("classes.names")) {
    args.classes = split_classes("classes.names");
  }
  if (!args.classes.empty()) {
    if (!conf_str.empty()) args.class_conf = parse_class_thresholds(conf_str, args.classes, args.score_threshold, "--class-conf");
    else args.class_conf.assign(args.classes.size(), args.score_threshold);
    if (!iou_str.empty()) args.class_iou = parse_class_thresholds(iou_str, args.classes, 1.0, "--class-iou");
    else args.class_iou.assign(args.classes.size(), 1.0);
  }
  return args;
}

// 将 OpenCV BGR 图像转换为平面格式（CHW）的 32 位浮点张量数据
std::vector<float> planar_float_bgr(const cv::Mat& image) {
  const size_t pixels = static_cast<size_t>(image.cols) * image.rows;
  std::vector<float> tensor(pixels * 3);
  for (int y = 0; y < image.rows; ++y) {
    const cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < image.cols; ++x) {
      const size_t index = static_cast<size_t>(y) * image.cols + x;
      tensor[index] = static_cast<float>(row[x][0]);
      tensor[pixels + index] = static_cast<float>(row[x][1]);
      tensor[pixels * 2 + index] = static_cast<float>(row[x][2]);
    }
  }
  return tensor;
}

// 根据类别 ID 与类别总数获取可视化渲染对应的颜色（基于 HSV 转 RGB）
cv::Scalar color_for(int class_id, size_t num_classes = 1) {
  return get_class_color(class_id, num_classes);
}

// 将模型输出的局部 mask 双线性插值还原到原始图像大小与对应目标边界框内
std::vector<uint8_t> paste_mask(const float* mask, int mask_height, int mask_width,
                                const std::array<float, 4>& box, int image_height,
                                int image_width) {
  std::vector<uint8_t> result(static_cast<size_t>(image_width) * image_height, 0);
  const double box_width = static_cast<double>(box[2] - box[0]);
  const double box_height = static_cast<double>(box[3] - box[1]);
  if (box_width <= 0.0 || box_height <= 0.0) return result;
  for (int y = 0; y < image_height; ++y) {
    const double source_y = ((y + 0.5 - box[1]) / box_height) * mask_height - 0.5;
    if (source_y <= -1.0 || source_y >= mask_height) continue;
    const int y0 = static_cast<int>(std::floor(source_y));
    const int y1 = y0 + 1;
    const double wy1 = source_y - y0;
    const double wy0 = 1.0 - wy1;
    for (int x = 0; x < image_width; ++x) {
      const double source_x = ((x + 0.5 - box[0]) / box_width) * mask_width - 0.5;
      if (source_x <= -1.0 || source_x >= mask_width) continue;
      const int x0 = static_cast<int>(std::floor(source_x));
      const int x1 = x0 + 1;
      const double wx1 = source_x - x0;
      const double wx0 = 1.0 - wx1;
      double value = 0.0;
      if (y0 >= 0 && y0 < mask_height && x0 >= 0 && x0 < mask_width)
        value += mask[static_cast<size_t>(y0) * mask_width + x0] * wy0 * wx0;
      if (y0 >= 0 && y0 < mask_height && x1 >= 0 && x1 < mask_width)
        value += mask[static_cast<size_t>(y0) * mask_width + x1] * wy0 * wx1;
      if (y1 >= 0 && y1 < mask_height && x0 >= 0 && x0 < mask_width)
        value += mask[static_cast<size_t>(y1) * mask_width + x0] * wy1 * wx0;
      if (y1 >= 0 && y1 < mask_height && x1 >= 0 && x1 < mask_width)
        value += mask[static_cast<size_t>(y1) * mask_width + x1] * wy1 * wx1;
      result[static_cast<size_t>(y) * image_width + x] = value >= 0.5 ? 1 : 0;
    }
  }
  return result;
}

// 利用图像二阶中心矩（Moments）计算分割掩码的主方向角度与几何质心
void calculate_mask_direction(Detection& detection, int width, int height) {
  const cv::Mat mask(height, width, CV_8UC1, detection.mask.data());
  const cv::Moments moments = cv::moments(mask, true);
  if (moments.m00 < 2.0) return;
  const double cx = moments.m10 / moments.m00;
  const double cy = moments.m01 / moments.m00;
  const double covariance_xx = moments.mu20;
  const double covariance_xy = moments.mu11;
  const double covariance_yy = moments.mu02;
  double angle = 0.5 * std::atan2(2.0 * covariance_xy, covariance_xx - covariance_yy);
  if (std::sin(angle) < 0.0) angle += CV_PI;
  detection.angle_degrees = std::fmod(angle * 180.0 / CV_PI + 180.0, 180.0);
  detection.center_x = cvRound(cx);
  detection.center_y = cvRound(cy);
}

// 在图像上绘制目标类别名称和置信度标签
void draw_label(cv::Mat& image, const Detection& detection,
                const std::vector<std::string>& classes) {
  std::ostringstream stream;
  stream << classes[static_cast<size_t>(detection.class_id)] << ' '
         << std::fixed << std::setprecision(2) << detection.score;
  const std::string text = stream.str();
  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const double scale = 0.5;
  const int thickness = 1;
  int baseline = 0;
  const cv::Size size = cv::getTextSize(text, font, scale, thickness, &baseline);
  int x = cvRound(detection.box[0]);
  int box_y = cvRound(detection.box[1]);
  x = std::max(0, std::min(x, image.cols - size.width - 4));
  box_y = std::max(size.height + baseline + 2, std::min(box_y, image.rows - 1));
  const int top = box_y - size.height - baseline - 2;
  const cv::Scalar color = color_for(detection.class_id, classes.size());
  cv::rectangle(image, cv::Point(x, top), cv::Point(x + size.width + 4, box_y),
                color, cv::FILLED);
  cv::putText(image, text, cv::Point(x + 2, box_y - baseline - 1), font, scale,
              cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
}

// 渲染可视化结果：绘制半透明分割掩码、检测矩形框、类别标签及主方向箭头
void render(cv::Mat& image, const std::vector<Detection>& detections,
            const std::vector<std::string>& classes) {
  const double alpha = 0.25;
  std::vector<std::pair<cv::Point, cv::Point> > arrows;
  for (const Detection& detection : detections) {
    const cv::Scalar color = color_for(detection.class_id, classes.size());
    const cv::Mat mask(image.rows, image.cols, CV_8UC1,
                       const_cast<uint8_t*>(detection.mask.data()));
    cv::Mat color_layer = image.clone();
    color_layer.setTo(color, mask);
    cv::addWeighted(color_layer, alpha, image, 1.0 - alpha, 0.0, image);
    cv::rectangle(image,
                  cv::Point(cvRound(detection.box[0]), cvRound(detection.box[1])),
                  cv::Point(cvRound(detection.box[2]), cvRound(detection.box[3])),
                  color, 2);
    draw_label(image, detection, classes);
    if (detection.class_id == 0 && std::isfinite(detection.angle_degrees)) {
      const double radians = detection.angle_degrees * CV_PI / 180.0;
      const double length = std::max(detection.box[2] - detection.box[0],
                                     detection.box[3] - detection.box[1]);
      const cv::Point start(detection.center_x, detection.center_y);
      const cv::Point end(cvRound(start.x + length * std::cos(radians)),
                          cvRound(start.y + length * std::sin(radians)));
      arrows.push_back(std::make_pair(start, end));
    }
  }
  for (const std::pair<cv::Point, cv::Point>& arrow : arrows)
    cv::arrowedLine(image, arrow.first, arrow.second, cv::Scalar(0, 0, 255),
                    4, cv::LINE_AA, 0, 0.25);
}

// 提取掩码外轮廓并序列化为多边形顶点坐标 JSON 数组
void write_mask_polygons(std::ostream& out, const std::vector<uint8_t>& binary, int height, int width) {
  cv::Mat mask(height, width, CV_8U, const_cast<uint8_t*>(binary.data()));
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  out << "[";
  for (size_t c = 0; c < contours.size(); ++c) {
    if (c) out << ", ";
    out << "[";
    for (size_t p = 0; p < contours[c].size(); ++p) {
      if (p) out << ", ";
      out << "[" << contours[c][p].x << ", " << contours[c][p].y << "]";
    }
    out << "]";
  }
  out << "]";
}

// 将推理预测结果导出至 JSON 文件（包含检测框、掩码面积、FNV-1a 哈希、RLE 与多边形坐标）
void write_json(const std::string& path, const std::string& image_path,
                const std::vector<Detection>& detections,
                const std::vector<std::string>& classes,
                int height, int width) {
  create_directories(parent_path(path));
  std::ofstream output(path.c_str());
  if (!output) throw std::runtime_error("Unable to write JSON: " + path);
  output << std::setprecision(9) << "{\n  \"image\": \"" << image_path
         << "\",\n  \"width\": " << width
         << ",\n  \"height\": " << height
         << ",\n  \"detections\": [\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const Detection& d = detections[i];
    output << "    {\"class_id\": " << d.class_id << ", \"class_name\": \""
           << classes[static_cast<size_t>(d.class_id)] << "\", \"score\": " << d.score
           << ", \"bbox_xyxy\": [" << d.box[0] << ", " << d.box[1] << ", "
           << d.box[2] << ", " << d.box[3] << "]";
    uint64_t mask_hash = 1469598103934665603ULL;
    size_t mask_area = 0;
    for (const uint8_t value : d.mask) {
      mask_area += value != 0;
      mask_hash ^= value;
      mask_hash *= 1099511628211ULL;
    }
    output << ", \"mask_area\": " << mask_area << ", \"mask_fnv1a64\": \""
           << std::hex << std::setw(16) << std::setfill('0') << mask_hash
           << std::dec << std::setfill(' ') << "\", \"mask_rle\": [";
    uint8_t current = 0;
    size_t run = 0;
    bool first_run = true;
    for (const uint8_t value : d.mask) {
      if (value == current) ++run;
      else {
        if (!first_run) output << ", ";
        output << run;
        first_run = false;
        current = value;
        run = 1;
      }
    }
    if (!first_run) output << ", ";
    output << run << "]";
    output << ", \"mask_polygons\": ";
    write_mask_polygons(output, d.mask, height, width);
    if (std::isfinite(d.angle_degrees))
      output << ", \"angle_degrees\": " << d.angle_degrees;
    output << "}" << (i + 1 == detections.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}

// 单张图像推理流程：图像读取校验、前向推理、后处理、NMS、结果存储与可视化
void infer_image(Ort::Session& session, const std::string& input_path,
                 const std::string& output_path,
                 const std::vector<std::string>& classes,
                 const std::vector<double>& class_conf,
                 const std::vector<double>& class_iou,
                 int expected_height, int expected_width,
                 bool draw = true) {
  cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::runtime_error("Unable to decode image: " + input_path +
                             " (OpenCV supports BMP, PNG, JPEG, TIFF and WebP)");
  if (image.cols != expected_width || image.rows != expected_height)
    throw std::runtime_error("The model input is fixed at " + std::to_string(expected_width) + "x" + std::to_string(expected_height) + "; received " +
                             std::to_string(image.cols) + "x" + std::to_string(image.rows) +
                             " for " + input_path +
                             ". Re-export the ONNX model with this image size; no implicit resize is applied.");
  std::vector<float> input_data = planar_float_bgr(image);
  const std::array<int64_t, 3> input_shape{{3, image.rows, image.cols}};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
  const std::array<const char*, 1> input_names{{"image"}};
  const std::array<const char*, 4> output_names{{"boxes", "scores", "classes", "mask_probs"}};
  const std::chrono::steady_clock::time_point infer_start = std::chrono::steady_clock::now();
  std::vector<Ort::Value> outputs = session.Run(
      Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
      output_names.data(), output_names.size());
  const std::chrono::steady_clock::time_point infer_end = std::chrono::steady_clock::now();
  std::cout << "Inference latency: "
            << std::chrono::duration<double, std::milli>(infer_end - infer_start).count()
            << " ms\n";

  const std::vector<int64_t> box_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<int64_t> score_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<int64_t> class_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<int64_t> mask_shape = outputs[3].GetTensorTypeAndShapeInfo().GetShape();
  if (box_shape.size() != 2 || score_shape.size() != 1 || class_shape.size() != 1 ||
      mask_shape.size() != 4 || box_shape[0] != score_shape[0] ||
      box_shape[0] != class_shape[0] || box_shape[0] != mask_shape[0] ||
      box_shape[1] != 4 || mask_shape[1] != 1)
    throw std::runtime_error("Unexpected ONNX output shape");
  const size_t count = static_cast<size_t>(box_shape[0]);
  const float* boxes = outputs[0].GetTensorData<float>();
  const float* scores = outputs[1].GetTensorData<float>();
  const int64_t* class_ids = outputs[2].GetTensorData<int64_t>();
  const float* masks = outputs[3].GetTensorData<float>();
  const int mask_height = static_cast<int>(mask_shape[2]);
  const int mask_width = static_cast<int>(mask_shape[3]);
  const size_t mask_pixels = static_cast<size_t>(mask_width) * mask_height;
  std::vector<std::string> eff_classes = classes;
  std::vector<double> eff_class_conf = class_conf;
  std::vector<double> eff_class_iou = class_iou;
  if (eff_classes.empty()) {
    int max_id = -1;
    for (size_t i = 0; i < count; ++i) {
      if (class_ids[i] > max_id) max_id = static_cast<int>(class_ids[i]);
    }
    const int num_cls = std::max(1, max_id + 1);
    for (int c = 0; c < num_cls; ++c) eff_classes.push_back("class_" + std::to_string(c));
    eff_class_conf.assign(eff_classes.size(), 0.5);
    eff_class_iou.assign(eff_classes.size(), 1.0);
  }
  std::vector<Detection> detections;
  for (size_t i = 0; i < count; ++i) {
    const int class_id = static_cast<int>(class_ids[i]);
    if (class_id < 0 || class_id >= static_cast<int>(eff_classes.size()) ||
        scores[i] < eff_class_conf[class_id]) continue;
    Detection detection;
    detection.class_id = class_id;
    detection.score = scores[i];
    std::copy_n(boxes + i * 4, 4, detection.box.begin());
    detection.mask = paste_mask(masks + i * mask_pixels, mask_height, mask_width,
                                detection.box, image.rows, image.cols);
    if (class_id == 0) calculate_mask_direction(detection, image.cols, image.rows);
    detections.push_back(detection);
  }
  detections = class_nms(std::move(detections), eff_class_conf, eff_class_iou);
  std::cout << "Valid detections after class-wise NMS: " << detections.size() << '\n';
  create_directories(parent_path(output_path));
  write_json(replace_extension(output_path, ".json"), input_path, detections, eff_classes, image.rows, image.cols);
  if (draw) {
    render(image, detections, eff_classes);
    write_image(output_path, image);
    std::cout << "Output: " << output_path << '\n';
  }
}

// 主函数：初始化环境、加载模型并执行推理
int main(int argc, char** argv) {
  RunLogger run_logger("CPP_INFER");
  try {
  const Arguments args = parse_arguments(argc, argv);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "instance_seg");
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetLogSeverityLevel(3);
  if (args.device == "cuda") {
#if defined(ENABLE_CUDA_PROVIDER) || defined(SWIN_ENABLE_CUDA)
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    options.AppendExecutionProvider_CUDA(cuda_options);
#else
    throw std::runtime_error("This executable was built without the CUDA provider");
#endif
  }

  const std::chrono::steady_clock::time_point load_start = std::chrono::steady_clock::now();
  Ort::Session session(env, args.model.c_str(), options);
  if (session.GetInputCount() != 1 || session.GetOutputCount() != 4)
    throw std::runtime_error(
        "Detectron2 Mask R-CNN ONNX requires one input and exactly four outputs: "
        "boxes, scores, classes, mask_probs");
  const std::vector<int64_t> input_shape =
      session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() != 3 || input_shape[0] != 3 || input_shape[1] <= 0 ||
      input_shape[2] <= 0)
    throw std::runtime_error("Detectron2 C++ contract requires fixed input image[3,H,W]");
  const int model_height = static_cast<int>(input_shape[1]);
  const int model_width = static_cast<int>(input_shape[2]);
  Ort::AllocatorWithDefaultOptions contract_allocator;
  const std::array<const char*, 4> contract_outputs{{"boxes", "scores", "classes", "mask_probs"}};
  for (size_t i = 0; i < contract_outputs.size(); ++i) {
    auto output_name = session.GetOutputNameAllocated(i, contract_allocator);
    if (output_name.get() == nullptr || std::string(output_name.get()) != contract_outputs[i])
      throw std::runtime_error("Detectron2 output names must be boxes,scores,classes,mask_probs");
  }
  const std::chrono::steady_clock::time_point load_end = std::chrono::steady_clock::now();
  std::cout << "OpenCV: " << CV_VERSION << '\n';
  std::cout << "Inference device: " << args.device << '\n';
  std::cout << "Model loading time: "
            << std::chrono::duration<double, std::milli>(load_end - load_start).count()
            << " ms\n";

  if (!args.serve) {
    const std::vector<std::string> inputs = image_paths(args.input);
    if (inputs.size() == 1 && fs::is_regular_file(args.input)) {
      infer_image(session, inputs.front(), args.output, args.classes, args.class_conf,
                  args.class_iou, model_height, model_width, args.draw);
    } else {
      create_directories(args.output);
      for (const std::string& input_path : inputs) {
        const fs::path input_file(input_path);
        const std::string output_path = join_path(
            args.output, input_file.stem().string() + ".png");
        infer_image(session, input_path, output_path, args.classes, args.class_conf,
                    args.class_iou, model_height, model_width, args.draw);
      }
    }
    return 0;
  }

  create_directories(args.output);
  if (!args.input.empty()) {
    infer_image(session, args.input, join_path(args.output, filename(args.input)),
                args.classes, args.class_conf, args.class_iou, model_height, model_width, args.draw);
  }
  std::cout << "Server mode started. Enter one image path per line or 'quit' to exit." << std::endl;
  std::string input_path;
  while (std::getline(std::cin, input_path)) {
    if (!input_path.empty() && input_path[input_path.size() - 1] == '\r') input_path.pop_back();
    if (input_path.empty()) continue;
    if (input_path == "quit" || input_path == "exit") break;
    try {
      infer_image(session, input_path, join_path(args.output, filename(input_path)),
                  args.classes, args.class_conf, args.class_iou, model_height, model_width);
    } catch (const std::exception& error) {
      std::cerr << "Image inference failed: " << error.what() << std::endl;
    }
  }
  std::cout << "Server mode stopped.\n";
  return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << "ONNX Runtime error: " << error.what() << '\n';
    return 2;
  } catch (const cv::Exception& error) {
    std::cerr << "OpenCV error: " << error.what() << '\n';
    return 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
