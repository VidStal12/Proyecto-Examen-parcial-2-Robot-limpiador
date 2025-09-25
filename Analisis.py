import numpy as np
import matplotlib.pyplot as plt
from control import tf, margin, freqresp
from control.matlab import ss, pole, zero
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# Crear ventana principal
root = tk.Tk()
root.title("Análisis del Sistema Linealizado del Robot Móvil")
root.geometry("1200x800")

# Configuración del sistema linealizado
d = 0.5  # Distancia del eje al punto de referencia
theta_ref = np.pi/4  # Orientación de referencia (45 grados)

# Matrices del sistema linealizado
A = np.zeros((3, 3))
B = np.array([[np.cos(theta_ref), -d*np.sin(theta_ref)],
              [np.sin(theta_ref), d*np.cos(theta_ref)],
              [0, 1]])
C = np.eye(3)
D = np.zeros((3, 2))

# Crear sistema de espacio de estados
sys = ss(A, B, C, D)

# Funciones de transferencia individuales
sys_vl_to_x = tf([B[0,0]], [1, 0])  # v_l -> x_R
sys_omega_to_x = tf([B[0,1]], [1, 0])  # omega -> x_R
sys_vl_to_y = tf([B[1,0]], [1, 0])  # v_l -> y_R
sys_omega_to_y = tf([B[1,1]], [1, 0])  # omega -> y_R
sys_omega_to_theta = tf([B[2,1]], [1, 0])  # omega -> theta

# Diccionario para mapear selecciones a sistemas
system_map = {
    "v_l → x_R": sys_vl_to_x,
    "ω → x_R": sys_omega_to_x,
    "v_l → y_R": sys_vl_to_y,
    "ω → y_R": sys_omega_to_y,
    "ω → θ": sys_omega_to_theta
}

# Función para realizar análisis de Bode y Nyquist
def realizar_analisis():
    # Obtener la función de transferencia seleccionada
    seleccion = combo_tf.get()
    
    if seleccion not in system_map:
        return
    
    sys_selected = system_map[seleccion]
    
    # Limpiar el frame de resultados
    for widget in frame_resultados.winfo_children():
        widget.destroy()
    
    # Crear figura con subplots
    fig = Figure(figsize=(10, 8))
    
    # 1. Diagrama de Polos y Ceros (estático)
    ax1 = fig.add_subplot(2, 2, 1)
    polos = pole(sys_selected)
    ceros = zero(sys_selected)
    
    # Dibujar polos y ceros
    ax1.plot(np.real(polos), np.imag(polos), 'rx', markersize=10, label='Polos')
    if len(ceros) > 0:
        ax1.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=10, label='Ceros')
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True)
    ax1.set_xlabel('Parte Real')
    ax1.set_ylabel('Parte Imaginaria')
    ax1.set_title('Diagrama de Polos y Ceros')
    ax1.legend()
    
    # Añadir información sobre estabilidad
    if all(np.real(polos) < 0):
        estabilidad = "Sistema Estable"
        color_est = 'green'
    elif any(np.real(polos) > 0):
        estabilidad = "Sistema Inestable"
        color_est = 'red'
    else:
        estabilidad = "Sistema Marginalmente Estable"
        color_est = 'orange'
    
    ax1.text(0.05, 0.95, f"Estabilidad: {estabilidad}", transform=ax1.transAxes, 
             color=color_est, fontweight='bold', verticalalignment='top')
    
    # 2. Diagrama de Bode (dinámico)
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Calcular respuesta en frecuencia usando freqresp en lugar de bode
    w = np.logspace(-2, 2, 1000)
    mag, phase, w = freqresp(sys_selected, w)
    
    # Convertir a dB
    mag_db = 20 * np.log10(mag)
    
    # Calcular márgenes de ganancia y fase
    try:
        gm, pm, wg, wp = margin(sys_selected)
        gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf
    except:
        gm_db, pm, wg, wp = np.inf, 0, 0, 0
    
    # Magnitud
    ax2.semilogx(w, mag_db.reshape(-1), 'b-')
    ax2.set_xlabel('Frecuencia [rad/s]')
    ax2.set_ylabel('Magnitud [dB]', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, which='both', axis='x')
    
    # Fase (eje secundario)
    ax2b = ax2.twinx()
    ax2b.semilogx(w, np.degrees(phase).reshape(-1), 'r-')
    ax2b.set_ylabel('Fase [grados]', color='r')
    ax2b.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Diagrama de Bode')
    
    # Añadir información de márgenes
    margin_text = f"Margen de Ganancia: {gm_db:.2f} dB\nMargen de Fase: {pm:.2f}°"
    if gm_db == np.inf:
        margin_text = f"Margen de Ganancia: ∞\nMargen de Fase: {pm:.2f}°"
    
    ax2.text(0.05, 0.95, margin_text, 
             transform=ax2.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Diagrama de Nyquist (dinámico)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Calcular diagrama de Nyquist manualmente
    nyquist_real = np.real(mag * np.exp(1j * phase)).reshape(-1)
    nyquist_imag = np.imag(mag * np.exp(1j * phase)).reshape(-1)
    
    ax3.plot(nyquist_real, nyquist_imag, 'b-')
    ax3.plot(nyquist_real, -nyquist_imag, 'r--')  # Simétrico
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(-1, color='red', linestyle='--', linewidth=1)  # Punto crítico
    ax3.grid(True)
    ax3.set_xlabel('Parte Real')
    ax3.set_ylabel('Parte Imaginaria')
    ax3.set_title('Diagrama de Nyquist')
    
    # 4. Información del sistema (estático)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    info_text = f"""
    Información del Sistema:
    Función de Transferencia: {seleccion}
    Polos: {polos}
    Ceros: {ceros}
    Margen de Ganancia: {gm_db:.2f} dB
    Margen de Fase: {pm:.2f}°
    Estabilidad: {estabilidad}
    
    Interpretación:
    - Polos en el origen indican un sistema integrador
    - Sistema marginalmente estable en lazo abierto
    - Se requiere realimentación para estabilizar el sistema
    """
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, verticalalignment='top', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Ajustar layout
    fig.tight_layout()
    
    # Mostrar figura en tkinter
    canvas = FigureCanvasTkAgg(fig, frame_resultados)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Crear interfaz
frame_control = ttk.Frame(root)
frame_control.pack(fill=tk.X, padx=10, pady=10)

ttk.Label(frame_control, text="Seleccionar Función de Transferencia:", 
          font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)

# Combo box para seleccionar la función de transferencia
tf_options = ["v_l → x_R", "ω → x_R", "v_l → y_R", "ω → y_R", "ω → θ"]
combo_tf = ttk.Combobox(frame_control, values=tf_options, state="readonly", width=15)
combo_tf.set("v_l → x_R")
combo_tf.pack(side=tk.LEFT, padx=5, pady=5)

# Botón para realizar análisis
btn_analizar = ttk.Button(frame_control, text="Realizar Análisis", command=realizar_analisis)
btn_analizar.pack(side=tk.LEFT, padx=5, pady=5)

# Añadir información sobre parámetros
param_text = f"Parámetros del sistema: d = {d}, θ_ref = {theta_ref:.2f} rad"
label_params = ttk.Label(frame_control, text=param_text, font=('Arial', 9))
label_params.pack(side=tk.LEFT, padx=20, pady=5)

# Frame para resultados
frame_resultados = ttk.Frame(root)
frame_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Realizar análisis inicial
realizar_analisis()

# Ejecutar la aplicación
root.mainloop()
