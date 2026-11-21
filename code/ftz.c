/* Switch flush-to-zero (FTZ) and denormals-are-zero (DAZ) on or off for the
   calling thread. x86-64 only. Build: gcc -O2 -shared -fPIC -o libftz.so ftz.c */
#include <xmmintrin.h>
void ftz_on(void)  { _mm_setcsr(_mm_getcsr() | 0x8040); }
void ftz_off(void) { _mm_setcsr(_mm_getcsr() & ~0x8040); }
