// Linux 平台 glibc 兼容性适配层：
// 通过符号版本重定向，将 __libc_start_main 绑定到 GLIBC_2.2.5，
// 从而确保编译生成的二进制文件可以在较低版本的 glibc 环境中正常加载与运行。

#if defined(__linux__)
extern "C" {

// 主函数指针类型定义
typedef int (*MainFunction)(int, char**, char**);

// 声明旧版本 glibc 的 __libc_start_main 入口函数原型
int compatible_libc_start_main(MainFunction main_function, int argc, char** argv,
                                void (*init)(), void (*fini)(), void (*rtld_fini)(),
                                void* stack_end);

// 使用汇编指令将 compatible_libc_start_main 链接至 GLIBC_2.2.5 符号
__asm__(".symver compatible_libc_start_main,__libc_start_main@GLIBC_2.2.5");

// __wrap___libc_start_main 包装函数：供链接器 --wrap 参数拦截调用
int __wrap___libc_start_main(MainFunction main_function, int argc, char** argv,
                             void (*init)(), void (*fini)(), void (*rtld_fini)(),
                             void* stack_end) {
  return compatible_libc_start_main(main_function, argc, argv, init, fini,
                                    rtld_fini, stack_end);
}

}
#endif
