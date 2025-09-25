import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from control import tf, margin, freqresp, step_response
from control.matlab import ss, pole, zero
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sympy as sp
import warnings
warnings.filterwarnings('ignore')

# Definir variables simbólicas
x, y, theta, v_l, omega, d = sp.symbols('x y theta v_l omega d')
t = sp.symbols('t')

class MetricasDesempeno:
    """Clase para calcular métricas de desempeño del sistema"""
    
    @staticmethod
    def calcular_metricas_step(respuesta, tiempo, referencia, señal_control=None, T=None):
        """
        Calcula métricas de desempeño a partir de la respuesta al escalón
        
        Parameters:
        respuesta (array): Respuesta del sistema
        tiempo (array): Vector de tiempo
        referencia (float): Valor de referencia
        señal_control (array): Señal de control (opcional)
        T (float): Horizonte de integración (opcional)
        
        Returns:
        dict: Diccionario con todas las métricas calculadas
        """
        # Si no se proporciona T, usar el tiempo final de simulación
        if T is None:
            T = tiempo[-1]
        
        # Encontrar índice correspondiente a T
        idx_T = np.argmax(tiempo >= T)
        if idx_T == 0:
            idx_T = len(tiempo) - 1
        
        # Recortar arrays al horizonte T
        tiempo_T = tiempo[:idx_T]
        respuesta_T = respuesta[:idx_T]
        
        # 1. Cálculo de overshoot (%)
        valor_final = respuesta_T[-1]
        maximo = np.max(respuesta_T)
        overshoot = 0
        if valor_final != 0:
            overshoot = ((maximo - valor_final) / valor_final) * 100
            if overshoot < 0:
                overshoot = 0
        
        # 2. Cálculo de settling time (s) - criterio del 2%
        valor_estable = valor_final
        margen = 0.02 * abs(valor_estable)
        
        # Encontrar cuándo la respuesta entra y permanece en el rango del 2%
        dentro_rango = np.abs(respuesta_T - valor_estable) <= margen
        settling_idx = None
        
        # Buscar el último punto donde la respuesta sale del rango
        for i in range(len(dentro_rango)-1, 0, -1):
            if not dentro_rango[i]:
                settling_idx = i + 1
                break
        
        if settling_idx is None and dentro_rango[0]:
            settling_time = 0
        else:
            settling_time = tiempo_T[settling_idx] if settling_idx is not None else tiempo_T[-1]
        
        # 3. Cálculo de steady state error
        steady_state_error = referencia - valor_final
        
        # 4. Cálculo de errores integrales
        error = referencia - respuesta_T
        
        # Error integral absoluto (IAE)
        iae = np.trapz(np.abs(error), tiempo_T)
        
        # Error integral absoluto ponderado por tiempo (ITAE)
        itae = np.trapz(tiempo_T * np.abs(error), tiempo_T)
        
        # 5. Cálculo de energía de la señal de control
        energia_control = 0
        norma_infinito = 0
        
        if señal_control is not None:
            señal_control_T = señal_control[:idx_T]
            # Energía (norma L2 al cuadrado)
            energia_control = np.trapz(señal_control_T**2, tiempo_T)
            
            # Norma infinito (valor máximo absoluto)
            norma_infinito = np.max(np.abs(señal_control_T))
        
        return {
            'overshoot_%': overshoot,
            'settling_time_s': settling_time,
            'steady_state_error': steady_state_error,
            'IAE': iae,
            'ITAE': itae,
            'energia_control': energia_control,
            'norma_infinito': norma_infinito,
            'tiempo_simulacion': T
        }
    
    @staticmethod
    def simular_sistema_controlado(sys, controlador, referencia, tiempo, estado_inicial=None):
        """
        Simula un sistema controlado con un controlador PID
        
        Parameters:
        sys: Sistema a controlar
        controlador: Función de transferencia del controlador
        referencia: Valor de referencia
        tiempo: Vector de tiempo
        estado_inicial: Estado inicial del sistema
        
        Returns:
        tuple: (respuesta, señal_control)
        """
        # Crear sistema en lazo cerrado
        sys_lazo_cerrado = feedback(controlador * sys, 1)
        
        # Simular respuesta al escalón
        t, respuesta = step_response(sys_lazo_cerrado, tiempo)
        
        # Calcular señal de control (simplificado)
        error = referencia - respuesta
        señal_control = np.zeros_like(tiempo)
        
        # Para un controlador PID, necesitaríamos implementar una simulación más detallada
        # Esta es una aproximación simplificada
        for i in range(1, len(tiempo)):
            dt = tiempo[i] - tiempo[i-1]
            # Aproximación simple de la acción de control
            señal_control[i] = señal_control[i-1] + error[i] * dt
        
        return respuesta, señal_control

class RobotMovilNoLineal:
    """Clase que representa el modelo no lineal del robot móvil"""
    
    def __init__(self, d_val=0.5):
        """
        Inicializa el modelo del robot móvil
        
        Parameters:
        d_val (float): Distancia del eje de las ruedas al punto de referencia
        """
        self.d = d_val
        self.equilibrium_points = None
        self.linear_matrices = None
        
        # Definir el modelo no lineal
        self.f1 = v_l * sp.cos(theta) - d * omega * sp.sin(theta)  # dx/dt
        self.f2 = v_l * sp.sin(theta) + d * omega * sp.cos(theta)  # dy/dt
        self.f3 = omega  # dtheta/dt
        
        # Vector de estado y entradas
        self.states = [x, y, theta]
        self.inputs = [v_l, omega]
    
    def encontrar_puntos_equilibrio(self):
        """
        Encuentra los puntos de equilibrio del sistema no lineal
        
        Returns:
        list: Lista de puntos de equilibrio
        """
        # Para encontrar equilibrio, igualamos las derivadas a cero
        eq1 = sp.Eq(self.f1, 0)
        eq2 = sp.Eq(self.f2, 0)
        eq3 = sp.Eq(self.f3, 0)
        
        # Resolver el sistema de ecuaciones
        soluciones = sp.solve([eq1, eq2, eq3], [v_l, omega], dict=True)
        
        # Los puntos de equilibrio son todos los puntos donde v_l=0 y omega=0
        # Cualquier x, y, theta es un punto de equilibrio con estas entradas
        self.equilibrium_points = soluciones
        
        return soluciones
    
    def linearizar(self, punto_equilibrio=None):
        """
        Lineariza el sistema no lineal alrededor de un punto de equilibrio
        
        Parameters:
        punto_equilibrio (dict): Punto de equilibrio alrededor del cual linearizar
        
        Returns:
        tuple: Matrices A y B del sistema linearizado
        """
        if punto_equilibrio is None:
            # Usar un punto de equilibrio por defecto (v_l=0, omega=0, theta=0)
            punto_equilibrio = {v_l: 0, omega: 0, theta: 0, x: 0, y: 0}
        
        # Calcular la matriz jacobiana respecto a los estados
        A = sp.Matrix([[sp.diff(self.f1, state) for state in self.states] 
                       for state in self.states])
        
        # Calcular la matriz jacobiana respecto a las entradas
        B = sp.Matrix([[sp.diff(self.f1, input_var) for input_var in self.inputs] 
                       for input_var in self.inputs])
        
        # Evaluar en el punto de equilibrio
        A_lin = A.subs(punto_equilibrio)
        B_lin = B.subs(punto_equilibrio)
        
        # Convertir a matrices numéricas
        A_np = np.array(A_lin).astype(np.float64)
        B_np = np.array(B_lin).astype(np.float64)
        
        self.linear_matrices = (A_np, B_np)
        
        return A_np, B_np
    
    def simular_no_lineal(self, estado_inicial, t_sim, entrada_func):
        """
        Simula el sistema no lineal
        
        Parameters:
        estado_inicial (list): Estado inicial [x0, y0, theta0]
        t_sim (array): Vector de tiempo para la simulación
        entrada_func (function): Función que devuelve las entradas [v_l, omega] en función del tiempo
        
        Returns:
        array: Estados del sistema a lo largo del tiempo
        """
        def modelo_no_lineal(estado, t):
            x_val, y_val, theta_val = estado
            v_l_val, omega_val = entrada_func(t)
            
            dxdt = v_l_val * np.cos(theta_val) - self.d * omega_val * np.sin(theta_val)
            dydt = v_l_val * np.sin(theta_val) + self.d * omega_val * np.cos(theta_val)
            dthetadt = omega_val
            
            return [dxdt, dydt, dthetadt]
        
        return odeint(modelo_no_lineal, estado_inicial, t_sim)

# Crear la aplicación principal
class AplicacionAnalisis:
    """Clase principal para la interfaz gráfica de análisis del robot móvil"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Robot Móvil No Lineal con Métricas de Desempeño")
        self.root.geometry("1400x900")
        
        # Crear el modelo del robot
        self.robot = RobotMovilNoLineal()
        
        # Encontrar puntos de equilibrio
        self.puntos_equilibrio = self.robot.encontrar_puntos_equilibrio()
        
        # Linearizar alrededor del primer punto de equilibrio
        self.A, self.B = self.robot.linearizar()
        
        # Configurar la interfaz
        self.configurar_interfaz()
        
        # Realizar análisis inicial
        self.mostrar_analisis_inicial()
    
    def configurar_interfaz(self):
        """Configura la interfaz gráfica de usuario"""
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Análisis básico
        self.tab_analisis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analisis, text="Análisis Básico")
        
        # Pestaña 2: Métricas de desempeño
        self.tab_metricas = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metricas, text="Métricas de Desempeño")
        
        # Configurar pestaña de análisis básico
        self.configurar_tab_analisis()
        
        # Configurar pestaña de métricas
        self.configurar_tab_metricas()
    
    def configurar_tab_analisis(self):
        """Configura la pestaña de análisis básico"""
        # Frame de control
        frame_control = ttk.Frame(self.tab_analisis)
        frame_control.pack(fill=tk.X, padx=10, pady=10)
        
        # Título
        titulo = ttk.Label(frame_control, text="Análisis de Robot Móvil No Lineal", 
                          font=('Arial', 14, 'bold'))
        titulo.pack(pady=10)
        
        # Información del modelo
        info_texto = f"""
        Modelo No Lineal:
        dx/dt = v_l * cos(θ) - d * ω * sin(θ)
        dy/dt = v_l * sin(θ) + d * ω * cos(θ)
        dθ/dt = ω
        
        Parámetros:
        d = {self.robot.d} (distancia del eje al punto de referencia)
        """
        info_label = ttk.Label(frame_control, text=info_texto, justify=tk.LEFT)
        info_label.pack(pady=5)
        
        # Botones de control
        frame_botones = ttk.Frame(frame_control)
        frame_botones.pack(pady=10)
        
        btn_puntos_eq = ttk.Button(frame_botones, text="Mostrar Puntos de Equilibrio", 
                                  command=self.mostrar_puntos_equilibrio)
        btn_puntos_eq.pack(side=tk.LEFT, padx=5)
        
        btn_matrices = ttk.Button(frame_botones, text="Mostrar Matrices Linealizadas", 
                                 command=self.mostrar_matrices_linealizadas)
        btn_matrices.pack(side=tk.LEFT, padx=5)
        
        btn_analisis = ttk.Button(frame_botones, text="Realizar Análisis Completo", 
                                 command=self.realizar_analisis_completo)
        btn_analisis.pack(side=tk.LEFT, padx=5)
        
        # Frame para resultados
        self.frame_resultados = ttk.Frame(self.tab_analisis)
        self.frame_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def configurar_tab_metricas(self):
        """Configura la pestaña de métricas de desempeño"""
        # Frame de control
        frame_control = ttk.Frame(self.tab_metricas)
        frame_control.pack(fill=tk.X, padx=10, pady=10)
        
        # Título
        titulo = ttk.Label(frame_control, text="Métricas de Desempeño", 
                          font=('Arial', 14, 'bold'))
        titulo.pack(pady=10)
        
        # Parámetros de simulación
        frame_params = ttk.Frame(frame_control)
        frame_params.pack(pady=10)
        
        ttk.Label(frame_params, text="Referencia:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_referencia = ttk.Entry(frame_params, width=10)
        self.entry_referencia.insert(0, "1.0")
        self.entry_referencia.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_params, text="Tiempo final (s):").grid(row=0, column=2, padx=5, pady=5)
        self.entry_t_final = ttk.Entry(frame_params, width=10)
        self.entry_t_final.insert(0, "10.0")
        self.entry_t_final.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(frame_params, text="Horizonte T (s):").grid(row=0, column=4, padx=5, pady=5)
        self.entry_horizonte = ttk.Entry(frame_params, width=10)
        self.entry_horizonte.insert(0, "10.0")
        self.entry_horizonte.grid(row=0, column=5, padx=5, pady=5)
        
        # Botón para calcular métricas
        btn_calcular = ttk.Button(frame_control, text="Calcular Métricas", 
                                 command=self.calcular_metricas)
        btn_calcular.pack(pady=10)
        
        # Frame para resultados de métricas
        self.frame_metricas = ttk.Frame(self.tab_metricas)
        self.frame_metricas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def mostrar_analisis_inicial(self):
        """Muestra el análisis inicial en la interfaz"""
        # Limpiar frame de resultados
        for widget in self.frame_resultados.winfo_children():
            widget.destroy()
        
        # Crear figura con subplots
        fig = Figure(figsize=(10, 8))
        
        # Información textual
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        info_text = f"""
        Análisis del Robot Móvil No Lineal
        
        Puntos de Equilibrio:
        El sistema tiene infinitos puntos de equilibrio donde:
        v_l = 0 y ω = 0
        Cualquier combinación de x, y, θ es un punto de equilibrio.
        
        Matrices del Sistema Linealizado:
        Matriz A (Estado):
        {self.A}
        
        Matriz B (Entrada):
        {self.B}
        
        Interpretación:
        - El sistema linealizado tiene polos en el origen (sistema integrador)
        - Es marginalmente estable en lazo abierto
        - Se requiere realimentación para estabilizar el sistema
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, verticalalignment='top', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Mostrar figura en tkinter
        canvas = FigureCanvasTkAgg(fig, self.frame_resultados)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def mostrar_puntos_equilibrio(self):
        """Muestra los puntos de equilibrio del sistema"""
        messagebox.showinfo("Puntos de Equilibrio", 
                           f"El sistema tiene los siguientes puntos de equilibrio:\n\n"
                           f"{self.puntos_equilibrio}\n\n"
                           f"Esto significa que cualquier combinación de x, y, θ es un punto de equilibrio\n"
                           f"siempre que v_l = 0 y ω = 0.")
    
    def mostrar_matrices_linealizadas(self):
        """Muestra las matrices del sistema linearizado"""
        messagebox.showinfo("Matrices Linealizadas", 
                           f"Matriz A (Estado):\n{self.A}\n\n"
                           f"Matriz B (Entrada):\n{self.B}\n\n"
                           f"El sistema linearizado es:\n"
                           f"dX/dt = A·X + B·U")
    
    def realizar_analisis_completo(self):
        """Realiza un análisis completo del sistema"""
        # Limpiar frame de resultados
        for widget in self.frame_resultados.winfo_children():
            widget.destroy()
        
        # Crear figura con subplots
        fig = Figure(figsize=(12, 10))
        
        # 1. Diagrama de Polos y Ceros del sistema linearizado
        ax1 = fig.add_subplot(2, 2, 1)
        
        # Crear sistema de espacio de estados
        C = np.eye(3)
        D = np.zeros((3, 2))
        sys = ss(self.A, self.B, C, D)
        
        # Obtener polos y ceros
        polos = pole(sys)
        ceros = zero(sys)
        
        # Dibujar polos y ceros
        ax1.plot(np.real(polos), np.imag(polos), 'rx', markersize=10, label='Polos')
        if len(ceros) > 0:
            ax1.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=10, label='Ceros')
        
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True)
        ax1.set_xlabel('Parte Real')
        ax1.set_ylabel('Parte Imaginaria')
        ax1.set_title('Diagrama de Polos y Ceros del Sistema Linealizado')
        ax1.legend()
        
        # 2. Simulación del sistema no lineal
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Definir función de entrada
        def entrada_func(t):
            if t < 2:
                return [0.0, 0.0]
            elif t < 5:
                return [0.5, 0.2]
            else:
                return [0.3, -0.1]
        
        # Simular sistema no lineal
        t_sim = np.linspace(0, 10, 1000)
        estado_inicial = [0.1, -0.1, 0.05]
        estados = self.robot.simular_no_lineal(estado_inicial, t_sim, entrada_func)
        
        # Graficar resultados
        ax2.plot(t_sim, estados[:, 0], 'r-', label='x')
        ax2.plot(t_sim, estados[:, 1], 'g-', label='y')
        ax2.plot(t_sim, estados[:, 2], 'b-', label='θ')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Estados')
        ax2.set_title('Simulación del Sistema No Lineal')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Diagrama de Bode (para la primera entrada y primera salida)
        ax3 = fig.add_subplot(2, 2, 3)
        
        # Crear función de transferencia v_l -> x
        sys_tf = tf([self.B[0,0]], [1, 0])
        
        # Calcular respuesta en frecuencia
        w = np.logspace(-2, 2, 1000)
        mag, phase, w = freqresp(sys_tf, w)
        
        # Convertir a dB
        mag_db = 20 * np.log10(mag)
        
        # Magnitud
        ax3.semilogx(w, mag_db.reshape(-1), 'b-')
        ax3.set_xlabel('Frecuencia [rad/s]')
        ax3.set_ylabel('Magnitud [dB]', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.grid(True, which='both', axis='x')
        
        # Fase (eje secundario)
        ax3b = ax3.twinx()
        ax3b.semilogx(w, np.degrees(phase).reshape(-1), 'r-')
        ax3b.set_ylabel('Fase [grados]', color='r')
        ax3b.tick_params(axis='y', labelcolor='r')
        
        ax3.set_title('Diagrama de Bode (v_l → x)')
        
        # 4. Diagrama de Nyquist
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Calcular diagrama de Nyquist
        nyquist_real = np.real(mag * np.exp(1j * phase)).reshape(-1)
        nyquist_imag = np.imag(mag * np.exp(1j * phase)).reshape(-1)
        
        ax4.plot(nyquist_real, nyquist_imag, 'b-')
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.axvline(-1, color='red', linestyle='--', linewidth=1)  # Punto crítico
        ax4.grid(True)
        ax4.set_xlabel('Parte Real')
        ax4.set_ylabel('Parte Imaginaria')
        ax4.set_title('Diagrama de Nyquist (v_l → x)')
        
        # Ajustar layout
        fig.tight_layout()
        
        # Mostrar figura en tkinter
        canvas = FigureCanvasTkAgg(fig, self.frame_resultados)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def calcular_metricas(self):
        """Calcula y muestra las métricas de desempeño"""
        try:
            # Obtener parámetros de la interfaz
            referencia = float(self.entry_referencia.get())
            t_final = float(self.entry_t_final.get())
            T = float(self.entry_horizonte.get())
            
            # Crear función de transferencia del sistema
            sys_tf = tf([self.B[0,0]], [1, 0])  # v_l -> x
            
            # Simular respuesta al escalón
            tiempo = np.linspace(0, t_final, 1000)
            t, respuesta = step_response(sys_tf, tiempo)
            
            # Crear una señal de control simplificada (para demostración)
            # En un caso real, esto vendría de un controlador
            señal_control = np.ones_like(tiempo) * 0.5
            
            # Calcular métricas
            metricas = MetricasDesempeno.calcular_metricas_step(
                respuesta, tiempo, referencia, señal_control, T
            )
            
            # Mostrar resultados
            self.mostrar_resultados_metricas(metricas, respuesta, tiempo, referencia, señal_control)
            
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos.")
    
    def mostrar_resultados_metricas(self, metricas, respuesta, tiempo, referencia, señal_control):
        """Muestra los resultados de las métricas de desempeño"""
        # Limpiar frame de métricas
        for widget in self.frame_metricas.winfo_children():
            widget.destroy()
        
        # Crear figura con subplots
        fig = Figure(figsize=(12, 10))
        
        # 1. Gráfica de la respuesta al escalón
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(tiempo, respuesta, 'b-', label='Respuesta')
        ax1.axhline(referencia, color='r', linestyle='--', label='Referencia')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Amplitud')
        ax1.set_title('Respuesta al Escalón')
        ax1.legend()
        ax1.grid(True)
        
        # Marcar settling time
        ax1.axvline(metricas['settling_time_s'], color='g', linestyle=':', 
                   label=f"Settling time: {metricas['settling_time_s']:.2f}s")
        ax1.legend()
        
        # 2. Gráfica de la señal de control
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(tiempo, señal_control, 'r-')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Señal de Control')
        ax2.set_title('Señal de Control')
        ax2.grid(True)
        
        # Marcar norma infinito
        ax2.axhline(metricas['norma_infinito'], color='g', linestyle=':', 
                   label=f"Norma ∞: {metricas['norma_infinito']:.2f}")
        ax2.legend()
        
        # 3. Gráfica de errores integrales
        ax3 = fig.add_subplot(2, 2, 3)
        error = referencia - respuesta
        error_abs = np.abs(error)
        
        # Calcular integrales acumulativas
        iae_acum = cumtrapz(error_abs, tiempo, initial=0)
        itae_acum = cumtrapz(tiempo * error_abs, tiempo, initial=0)
        
        ax3.plot(tiempo, iae_acum, 'b-', label='IAE acumulado')
        ax3.plot(tiempo, itae_acum, 'r-', label='ITAE acumulado')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Error Integral')
        ax3.set_title('Errores Integrales Acumulados')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Tabla de métricas
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        metricas_texto = f"""
        Métricas de Desempeño:
        
        Overshoot: {metricas['overshoot_%']:.2f}%
        Settling time: {metricas['settling_time_s']:.2f} s
        Steady state error: {metricas['steady_state_error']:.4f}
        
        IAE: {metricas['IAE']:.4f}
        ITAE: {metricas['ITAE']:.4f}
        
        Energía de control: {metricas['energia_control']:.4f}
        Norma ∞: {metricas['norma_infinito']:.4f}
        
        Horizonte de integración: {metricas['tiempo_simulacion']} s
        """
        
        ax4.text(0.05, 0.95, metricas_texto, transform=ax4.transAxes, verticalalignment='top', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Ajustar layout
        fig.tight_layout()
        
        # Mostrar figura en tkinter
        canvas = FigureCanvasTkAgg(fig, self.frame_metricas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionAnalisis(root)
    root.mainloop()
