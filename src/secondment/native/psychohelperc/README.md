# Native psychoacoustic backend

These sources are the sharpness and tonality subset of the METAVISION
secondment analysis backend supplied by the project maintainer. The Streamlit
app builds them into one source-versioned shared library at runtime.

Included calculation path:

- `ISO_532-1.c/.h`: ISO 532-1 time-varying loudness and specific loudness.
- `tonality_aures1985.c/.h`: Aures (1985) time-varying tonality and K5.
- `pocketfft.c/.h`: FFT implementation used by the tonality calculation.

The unrelated roughness, CLI, WAV I/O, and OpenMP sources are intentionally
excluded. PocketFFT is redistributed under its BSD 3-Clause license in
`LICENSE.pocketfft.md`.

Typical builds:

```sh
# Linux
cc -O3 -std=c99 -DISO532_BUILD -shared -fPIC \
  -o libmetavision_psychoacoustics.so \
  ISO_532-1.c tonality_aures1985.c pocketfft.c -lm

# Windows / MinGW
gcc -O3 -std=c99 -DISO532_BUILD -shared -static-libgcc \
  -Wl,--export-all-symbols -o metavision_psychoacoustics.dll \
  ISO_532-1.c tonality_aures1985.c pocketfft.c -lm
```
