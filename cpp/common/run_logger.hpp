#pragma once

#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>

namespace fs = std::filesystem;

// 跨平台本地时间获取辅助函数（兼容 Linux 的 localtime_r 与 Windows MSVC 的 localtime_s）
inline std::tm safe_localtime(const std::time_t& time_val) {
  std::tm local_time{};
#if defined(_WIN32)
  localtime_s(&local_time, &time_val);
#else
  localtime_r(&time_val, &local_time);
#endif
  return local_time;
}

// 自定义流缓冲区（streambuf）：实现双向输出重定向（Tee），
// 将数据同时写入终端与日志文件，并在每行开头自动添加时间戳与模块标签。
class PrefixTeeBuffer : public std::streambuf {
 public:
  PrefixTeeBuffer(std::streambuf* terminal, std::streambuf* log, const std::string& module)
      : terminal_(terminal), log_(log), module_(module) {}

 protected:
  // 重写溢出处理函数：在写入新行首个字符前追加时间戳及模块前缀，随后写入字符
  int overflow(int value) override {
    if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
    const char character = traits_type::to_char_type(value);
    if (line_start_ && character != '\n' && character != '\r') {
      const auto now = std::chrono::system_clock::now();
      const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
      const std::tm local_time = safe_localtime(now_time);
      char ts[64];
      std::strftime(ts, sizeof(ts), "[%Y-%m-%d %H:%M:%S] ", &local_time);
      std::string prefix = std::string(ts) + "[" + module_ + "] ";
      terminal_->sputn(prefix.data(), static_cast<std::streamsize>(prefix.size()));
      log_->sputn(prefix.data(), static_cast<std::streamsize>(prefix.size()));
      line_start_ = false;
    }
    terminal_->sputc(character);
    log_->sputc(character);
    if (character == '\n') line_start_ = true;
    return value;
  }

  // 同步终端与日志流缓冲区
  int sync() override {
    return terminal_->pubsync() == 0 && log_->pubsync() == 0 ? 0 : -1;
  }

 private:
  std::streambuf* terminal_;
  std::streambuf* log_;
  std::string module_;
  bool line_start_ = true;
};

// 运行日志管理器：接管 std::cout 与 std::cerr 的流缓冲并保存至日志文件，在对象析构时自动还原
class RunLogger {
 public:
  explicit RunLogger(const std::string& module)
      : log_path_(timestamp_log_path(module)),
        log_(log_path_.c_str(), std::ios::app),
        stdout_buffer_(std::cout.rdbuf(), log_.rdbuf(), module),
        stderr_buffer_(std::cerr.rdbuf(), log_.rdbuf(), module),
        original_stdout_(std::cout.rdbuf(&stdout_buffer_)),
        original_stderr_(std::cerr.rdbuf(&stderr_buffer_)) {
    std::cout << "Logging initialized. Log file: " << log_path_ << '\n';
  }

  ~RunLogger() {
    std::cout.flush();
    std::cerr.flush();
    std::cout.rdbuf(original_stdout_);
    std::cerr.rdbuf(original_stderr_);
  }

 private:
  // 生成按当前日期和时间命名的日志文件路径（保存于 logs/YYYY-MM-DD/cpp_<module>_<HHMMSS>.log）
  static std::string timestamp_log_path(const std::string& module) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    const std::tm local_time = safe_localtime(now_time);
    char date_str[32];
    std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", &local_time);
    char time_str[32];
    std::strftime(time_str, sizeof(time_str), "%H%M%S", &local_time);

    fs::path dir = fs::path("logs") / date_str;
    fs::create_directories(dir);
    fs::path p = dir / ("cpp_" + module + "_" + std::string(time_str) + ".log");
    return p.string();
  }

  std::string log_path_;
  std::ofstream log_;
  PrefixTeeBuffer stdout_buffer_;
  PrefixTeeBuffer stderr_buffer_;
  std::streambuf* original_stdout_;
  std::streambuf* original_stderr_;
};
