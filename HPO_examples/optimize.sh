#!/bin/bash
 
# Ejecutar el script de optimización en un hilo separado
python3 ./optimizer.py &

# Obtener el PID del hilo de optimización
optimization_pid=$!

# Esperar un tiempo suficiente para que la optimización se lleve a cabo (puedes ajustar este tiempo según tu necesidad)
sleep 60

# Matar el hilo de optimización
kill $optimization_pid

# Llamar al script de selección y ejecución con la mejor configuración
python3 select_best_incumbent.py