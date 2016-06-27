#!/bin/bash
export PATH=/Soft/cuda/7.5.18/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N cuda-filters 
# Cambiar el shell
#$ -S /bin/bash

nvprof ./singleCardAsyn.exe images/radiation.png -f 6
nvprof ./singleCardAsyn.exe images/lena.png -f 6
mv images/lena_filter.png images/lena_asyn_filter.png
nvprof ./singleCardSyn.exe images/lena.png -f 6 -p
mv images/lena_filter.png images/lena_syn_filter.png
