// compat/error.h: Apple Clang / BSD-libc replacement for the GNU error.h API.
//
// The vendored solver calls error(status, errnum, format, ...) at fatal I/O
// points. GNU's implementation prints "prog: format: strerror(errnum)" to
// stderr and exits with `status`. This shim reproduces that contract with
// only C++17 standard library plus errno, so search semantics are untouched:
// every call site still terminates the process with the same status code.
#ifndef ALEX1770_WORDLE_COMPAT_ERROR_H
#define ALEX1770_WORDLE_COMPAT_ERROR_H

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

inline void error(int status, int errnum, const char* format, ...) {
  va_list args;
  va_start(args, format);
  std::fprintf(stderr, "%s: ", "wordle");
  std::vfprintf(stderr, format, args);
  va_end(args);
  if (errnum != 0) {
    std::fprintf(stderr, ": %s", std::strerror(errnum));
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);
  if (status != 0) {
    std::exit(status);
  }
}

#endif  // ALEX1770_WORDLE_COMPAT_ERROR_H
