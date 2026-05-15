# ╔══════════════════════════════════════════════════════════════╗
# ║   FainWatch — Detector de Lipotimia                         ║
# ║   Ejecutar: python3 fainwatch_local.py                      ║
# ║   Parar:    pulsa Q en la ventana de la cámara              ║
# ╚══════════════════════════════════════════════════════════════╝

import subprocess, sys, os

# --------------------------------------------------------------
# Instalación automática de librerías necesarias.
# Solo ocurre la primera vez — después ya están en el sistema.
# --------------------------------------------------------------
for pkg in ["mediapipe", "opencv-python", "Pillow", "reportlab"]:
    try:
        __import__(pkg.replace("-python","").replace("-","_"))
    except ImportError:
        print(f"Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# --------------------------------------------------------------
# Descarga automática de los modelos de MediaPipe.
# face_landmarker: detecta cara, ojos, boca, expresiones.
# pose_landmarker: detecta hombros y postura corporal.
# Solo se descargan si no existen ya en la carpeta.
# --------------------------------------------------------------
import urllib.request

MODELOS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}
for nombre, url in MODELOS.items():
    if not os.path.exists(nombre):
        print(f"Descargando {nombre}...")
        urllib.request.urlretrieve(url, nombre)
        print(f"✔ {nombre} listo")

# --------------------------------------------------------------
# Importaciones de todas las librerías del proyecto.
# --------------------------------------------------------------
import cv2                          # captura de cámara y visualización
import mediapipe as mp              # análisis facial y de postura
import numpy as np                  # cálculos matemáticos
import time                         # timestamps y medición de fps
import csv                          # escritura del histórico CSV
from collections import deque       # ventana deslizante de niveles de riesgo
from dataclasses import dataclass, field  # definición de estructuras de datos
from typing import Optional         # tipos opcionales en Config

from reportlab.lib.pagesizes import A4          # tamaño de página del PDF
from reportlab.lib import colors                # colores del PDF
from reportlab.lib.units import cm              # unidades del PDF
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("✔ Todo listo")

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# Todos los umbrales y parámetros del sistema en un solo lugar.
# Modifica aquí para ajustar el comportamiento del detector.
# ══════════════════════════════════════════════════════════════
@dataclass
class Config:
    umbral_blink:           float = 0.65   # score de cierre de ojos para considerar somnolencia
    umbral_boca:            float = 0.07   # apertura mínima de boca para activar la señal
    umbral_roll:            float = 20.0   # grados de inclinación lateral de cabeza
    umbral_pitch:           float = 25.0   # grados de caída hacia delante de la cabeza
    umbral_caida:           float = 0.10   # distancia nariz-hombros para caída postural
    umbral_palidez:         float = 0.25   # desviación de color respecto al baseline del donante
    umbral_malestar:        float = 0.30   # score de expresión de malestar (0-1)
    ear_margen_sigma:       float = 2.0    # desviaciones típicas sobre el EAR normal para mirada fija
    umbral_movimiento_iris: float = 0.008  # movimiento mínimo de iris para considerar iris inmóvil
    frames_mirada_fija:     int   = 22     # frames consecutivos para confirmar mirada fija (~1.8s)
    umbral_mar_bostezo:     float = 0.45   # apertura de boca mínima para considerar bostezo
    umbral_squint_bostezo:  float = 0.25   # entrecerrado de ojos mínimo para confirmar bostezo
    frames_bostezo:         int   = 14     # frames consecutivos para confirmar bostezo (~1.2s)
    umbral_velocidad_giro:  float = 3.5    # grados/frame; por encima se considera movimiento voluntario
    min_params_coherencia:  int   = 2      # mínimo de señales distintas activas para subir de nivel
    frames_postural_minimo: int   = 8      # frames sostenidos antes de que postura sume puntos
    calibracion_sigma:      float = 2.5    # desviaciones típicas para calcular umbrales personales
    alpha_ema:              float = 0.30   # suavizado exponencial de señales (0=sin suavizar, 1=máximo)
    frames_ojos_umbral:     int   = 10     # frames con ojos cerrados antes de activar la alerta
    ventana_riesgo:         int   = 15     # frames que promedia para calcular el nivel de riesgo
    frames_calibracion:     int   = 50     # frames iniciales para aprender los valores del donante
    num_faces:              int   = 1      # número de caras a detectar simultáneamente
    min_confianza:          float = 0.5    # confianza mínima de MediaPipe para aceptar detección
    camara_idx:             int   = 0      # índice de cámara (0=por defecto, 1=segunda cámara...)

# ══════════════════════════════════════════════════════════════
# LANDMARKS
# Índices de los puntos de referencia del rostro y cuerpo
# que usa MediaPipe internamente. No modificar.
# ══════════════════════════════════════════════════════════════
BOCA_SUP=13; BOCA_INF=14; BOCA_IZQ=61; BOCA_DER=291   # puntos del contorno de la boca
FRENTE=10;   MENTON=152;  NARIZ_TIP=1                  # puntos del eje vertical de la cara
MEJILLA_IZQ=234; MEJILLA_DER=454                       # puntos de las mejillas para análisis de piel
POSE_NARIZ=0; POSE_HOMBRO_IZQ=11; POSE_HOMBRO_DER=12  # puntos de postura corporal
OJO_IZQ_SUP=159; OJO_IZQ_INF=145; OJO_IZQ_EXT=33; OJO_IZQ_INT=133   # párpados ojo izquierdo
OJO_DER_SUP=386; OJO_DER_INF=374; OJO_DER_EXT=362; OJO_DER_INT=263   # párpados ojo derecho
IRIS_IZQ_CENTRO=468; IRIS_DER_CENTRO=473               # centros del iris para detectar mirada fija
ZONAS_PIEL = [(FRENTE,0.50),(MEJILLA_IZQ,0.25),(MEJILLA_DER,0.25)]    # zonas y pesos para palidez

# ══════════════════════════════════════════════════════════════
# BLENDSHAPES DE MALESTAR
# Gestos faciales involuntarios que indican malestar o dolor.
# Cada uno tiene un peso según su relevancia clínica.
# ══════════════════════════════════════════════════════════════
BLENDSHAPES_MALESTAR = {
    "browLowererLeft":0.15,   # ceño fruncido izquierdo — tensión o dolor
    "browLowererRight":0.15,  # ceño fruncido derecho
    "browInnerUp":0.10,       # ceja interior levantada — preocupación
    "noseSneerLeft":0.10,     # arrugar nariz izquierda — repulsión o malestar
    "noseSneerRight":0.10,    # arrugar nariz derecha
    "cheekSquintLeft":0.10,   # apretar mejilla izquierda — dolor o mueca
    "cheekSquintRight":0.10,  # apretar mejilla derecha
    "mouthFrownLeft":0.10,    # comisura izquierda hacia abajo — malestar
    "mouthFrownRight":0.10,   # comisura derecha hacia abajo
    "eyeSquintLeft":0.05,     # ojo izquierdo entrecerrado — dolor
    "eyeSquintRight":0.05,    # ojo derecho entrecerrado
}
_PESO_MALESTAR = sum(BLENDSHAPES_MALESTAR.values())

# ══════════════════════════════════════════════════════════════
# FILTRO EMA (Exponential Moving Average)
# Suaviza las señales para evitar alertas por movimientos
# bruscos puntuales. Alpha controla la velocidad de respuesta:
# valor bajo = más suave, valor alto = más reactivo.
# ══════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha; self.value = None

    def update(self, x):
        # Promedio ponderado entre valor actual y nuevo
        self.value = x if self.value is None else self.alpha*x+(1-self.alpha)*self.value
        return self.value

    def reset(self):
        self.value = None

# ══════════════════════════════════════════════════════════════
# BASELINE DE COLOR
# Aprende el color de piel de cada donante durante la
# calibración (frente y mejillas en espacio HSV).
# Luego detecta la palidez como desviación relativa a ese
# color base personal — funciona con cualquier tono de piel.
# También aprende la apertura ocular normal (EAR) del donante.
# ══════════════════════════════════════════════════════════════
class BaselineColor:
    def __init__(self):
        self._buf_V={idx:[] for idx,_ in ZONAS_PIEL}  # brillo (V en HSV)
        self._buf_S={idx:[] for idx,_ in ZONAS_PIEL}  # saturación (S en HSV)
        self._buf_G={idx:[] for idx,_ in ZONAS_PIEL}  # canal verde (sensible a perfusión)
        self.ref_V={}; self.ref_S={}; self.ref_G={}   # valores de referencia fijados
        self._buf_ear=[]                               # apertura ocular durante calibración
        self.ear_media=0.30; self.ear_std=0.03         # EAR de referencia personal
        self.listo=False

    def _stats(self, frame, lm, idx, h, w, tam=22):
        # Extrae un parche de piel alrededor del landmark y calcula sus estadísticas
        px=int(lm[idx].x*w); py=int(lm[idx].y*h)
        x1=max(0,px-tam//2); y1=max(0,py-tam//2)
        roi=frame[y1:min(h,y1+tam), x1:min(w,x1+tam)]
        if roi.size==0: return None
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV).astype(float)
        return hsv[:,:,2].mean(), hsv[:,:,1].mean(), roi[:,:,1].astype(float).mean()

    def acumular(self, frame, lm, ear=None):
        # Recoge muestras de color y EAR durante la calibración
        if self.listo: return
        h,w=frame.shape[:2]
        for idx,_ in ZONAS_PIEL:
            s=self._stats(frame,lm,idx,h,w)
            if s: V,S,G=s; self._buf_V[idx].append(V); self._buf_S[idx].append(S); self._buf_G[idx].append(G)
        if ear and ear>0.05: self._buf_ear.append(ear)  # ignora frames con ojos cerrados

    def fijar(self):
        # Calcula y guarda los valores de referencia al terminar la calibración
        for idx,_ in ZONAS_PIEL:
            if self._buf_V[idx]:
                self.ref_V[idx]=float(np.mean(self._buf_V[idx]))
                self.ref_S[idx]=float(np.mean(self._buf_S[idx]))
                self.ref_G[idx]=float(np.mean(self._buf_G[idx]))
        if self._buf_ear:
            self.ear_media=float(np.mean(self._buf_ear))
            self.ear_std=float(max(np.std(self._buf_ear),0.01))
        self.listo=True
        print(f"✔ Baseline fijado — EAR={self.ear_media:.3f} σ={self.ear_std:.3f}")

    def score_palidez(self, frame, lm):
        # Calcula cuánto se ha alejado el color actual del color base del donante
        # Combina: blanqueamiento (V sube), pérdida de color (S baja), caída de perfusión (G baja)
        if not self.listo: return 0.0
        h,w=frame.shape[:2]; total=peso=0.0
        for idx,pz in ZONAS_PIEL:
            if idx not in self.ref_V: continue
            s=self._stats(frame,lm,idx,h,w)
            if not s: continue
            V,S,G=s
            dV=max(0,(V-self.ref_V[idx])/255)   # sube si la piel se vuelve más blanca
            dS=max(0,-(S-self.ref_S[idx])/255)  # sube si la piel pierde color rosado
            dG=max(0,-(G-self.ref_G[idx])/255)  # sube si cae el canal verde (perfusión)
            total+=(0.45*dV+0.35*dS+0.20*dG)*pz; peso+=pz
        return float(np.clip(total/(peso*0.25),0,1)) if peso else 0.0

# ══════════════════════════════════════════════════════════════
# DETECTOR DE MIRADA FIJA
# Detecta la señal clínica de alarma vagal previa al síncope:
# ojos muy abiertos con iris inmóvil durante varios segundos.
# Usa el EAR personal del donante como umbral, no un valor fijo,
# para evitar falsos positivos con personas de ojos grandes.
# ══════════════════════════════════════════════════════════════
class DetectorMiradaFija:
    def __init__(self, cfg, bl):
        self.cfg=cfg; self.bl=bl
        self._pizq=None; self._pder=None  # posición anterior del iris
        self._frames=0                     # contador de frames con mirada fija
        self._ear=EMA(0.4); self._mov=EMA(0.4)

    def _ear_lm(self, lm, s, i, e, n):
        # EAR = distancia vertical párpados / distancia horizontal párpados
        v=np.sqrt((lm[s].x-lm[i].x)**2+(lm[s].y-lm[i].y)**2)
        h=np.sqrt((lm[e].x-lm[n].x)**2+(lm[e].y-lm[n].y)**2)
        return v/(h+1e-6)

    def actualizar(self, lm):
        # Calcula EAR actual y movimiento del iris respecto al frame anterior
        ei=self._ear_lm(lm,OJO_IZQ_SUP,OJO_IZQ_INF,OJO_IZQ_EXT,OJO_IZQ_INT)
        ed=self._ear_lm(lm,OJO_DER_SUP,OJO_DER_INF,OJO_DER_EXT,OJO_DER_INT)
        ear=self._ear.update((ei+ed)/2)
        ix,iy=lm[IRIS_IZQ_CENTRO].x,lm[IRIS_IZQ_CENTRO].y
        dx,dy=lm[IRIS_DER_CENTRO].x,lm[IRIS_DER_CENTRO].y
        if self._pizq is None:
            self._pizq=(ix,iy); self._pder=(dx,dy); return ear,0.0,False
        mi=np.sqrt((ix-self._pizq[0])**2+(iy-self._pizq[1])**2)
        md=np.sqrt((dx-self._pder[0])**2+(dy-self._pder[1])**2)
        self._pizq=(ix,iy); self._pder=(dx,dy)
        mov=self._mov.update((mi+md)/2)
        # Umbral personal: EAR normal del donante + N desviaciones típicas
        umbral=self.bl.ear_media+self.cfg.ear_margen_sigma*self.bl.ear_std
        if ear>umbral and mov<self.cfg.umbral_movimiento_iris: self._frames+=1
        else: self._frames=max(0,self._frames-1)  # decaimiento suave
        return ear, mov, self._frames>=self.cfg.frames_mirada_fija

# ══════════════════════════════════════════════════════════════
# DETECTOR DE BOSTEZO
# Detecta bostezos combinando boca muy abierta (MAR alto)
# con ojos entrecerrados simultáneamente (reflejo involuntario).
# Se registra en el historial pero NO afecta el nivel de riesgo.
# ══════════════════════════════════════════════════════════════
class DetectorBostezo:
    def __init__(self, cfg):
        self.cfg=cfg; self._f=0; self._activo=False
        self.total=0; self.ultimo="—"  # contador acumulado y hora del último

    def actualizar(self, lm, fr):
        if lm is None or not fr.face_blendshapes:
            self._f=max(0,self._f-1); return False
        # MAR: ratio apertura vertical / horizontal de la boca
        mar=dist(lm[BOCA_SUP],lm[BOCA_INF])/(dist(lm[BOCA_IZQ],lm[BOCA_DER])+1e-6)
        bl={b.category_name:b.score for b in fr.face_blendshapes[0]}
        sq=(bl.get("eyeSquintLeft",0)+bl.get("eyeSquintRight",0))/2
        if mar>self.cfg.umbral_mar_bostezo and sq>self.cfg.umbral_squint_bostezo: self._f+=1
        else: self._f=max(0,self._f-1)
        ok=self._f>=self.cfg.frames_bostezo
        if ok and not self._activo:
            # Flanco de subida: cuenta el bostezo una sola vez aunque dure varios segundos
            self._activo=True; self.total+=1; self.ultimo=time.strftime("%H:%M:%S")
        elif not ok: self._activo=False
        return ok

# ══════════════════════════════════════════════════════════════
# FILTRO DE MOVIMIENTO VOLUNTARIO
# Distingue movimiento voluntario (hablar, girar a mirar algo)
# de síntoma de lipotimia mediante tres estrategias:
#   1. Velocidad: giros rápidos = voluntarios, se ignoran
#   2. Duración: solo puntúa si el estado se mantiene varios frames
#   3. Coherencia: exige varias señales simultáneas (en DetectorLipotimia)
# ══════════════════════════════════════════════════════════════
class FiltroMovimientoVoluntario:
    def __init__(self, cfg):
        self.cfg=cfg
        self._rp=None; self._pp=None      # ángulos del frame anterior
        self._fr=0; self._fp=0; self._fc=0  # contadores de duración por parámetro

    def actualizar(self, roll, pitch, caida):
        c=self.cfg; rapido=False
        if self._rp is not None:
            # Si el cambio de ángulo entre frames es grande, es un giro voluntario
            if abs(roll-self._rp)>c.umbral_velocidad_giro or abs(pitch-self._pp)>c.umbral_velocidad_giro:
                rapido=True
        self._rp=roll; self._pp=pitch
        if rapido:
            # Movimiento voluntario: decrementar contadores y descartar este frame
            self._fr=max(0,self._fr-1); self._fp=max(0,self._fp-1); self._fc=max(0,self._fc-1)
            return False,False,False,True
        # Acumular duración: solo puntúa si se mantiene suficientes frames seguidos
        self._fr=self._fr+1 if roll>c.umbral_roll else max(0,self._fr-1)
        self._fp=self._fp+1 if pitch>c.umbral_pitch else max(0,self._fp-1)
        self._fc=self._fc+1 if caida>c.umbral_caida else max(0,self._fc-1)
        return self._fr>=c.frames_postural_minimo, self._fp>=c.frames_postural_minimo, self._fc>=c.frames_postural_minimo, False

# ══════════════════════════════════════════════════════════════
# CALIBRADOR
# Durante los primeros N frames aprende los valores normales
# del donante y ajusta los umbrales a esa persona concreta.
# Una vez calibrado no vuelve a modificar los umbrales.
# ══════════════════════════════════════════════════════════════
class Calibrador:
    def __init__(self, cfg, bl):
        self.cfg=cfg; self.bl=bl
        self.buf={k:[] for k in ["blink","boca","roll","pitch","caida","malestar"]}
        self.calibrado=False

    def agregar(self, frame, lm, blink, boca, roll, pitch, caida, malestar, ear=0.0):
        if self.calibrado: return
        if lm is not None: self.bl.acumular(frame,lm,ear=ear)
        for k,v in zip(self.buf,[blink,boca,roll,pitch,caida,malestar]):
            self.buf[k].append(v)
        if len(self.buf["blink"])>=self.cfg.frames_calibracion: self._calc()

    def _calc(self):
        # Umbral personal = media + N desviaciones típicas de cada parámetro
        k=self.cfg.calibracion_sigma
        for cl,at in [("boca","umbral_boca"),("roll","umbral_roll"),("pitch","umbral_pitch"),
                      ("caida","umbral_caida"),("malestar","umbral_malestar")]:
            d=np.array(self.buf[cl])
            setattr(self.cfg,at,max(float(np.mean(d)+k*np.std(d)),getattr(self.cfg,at)*0.5))
        self.bl.fijar(); self.calibrado=True
        print(f"✔ Calibración completa")

# ══════════════════════════════════════════════════════════════
# MODELOS MEDIAPIPE
# Carga los dos modelos de análisis:
# - FaceLandmarker: 478 puntos del rostro + blendshapes
# - PoseLandmarker: puntos del cuerpo para postura y caída
# ══════════════════════════════════════════════════════════════
class ModelosMP:
    def __init__(self, cfg):
        fo=vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=cfg.num_faces, output_face_blendshapes=True,
            min_face_detection_confidence=cfg.min_confianza,
            min_face_presence_confidence=cfg.min_confianza,
            min_tracking_confidence=cfg.min_confianza,
        )
        po=vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=cfg.min_confianza,
            min_pose_presence_confidence=cfg.min_confianza,
            min_tracking_confidence=cfg.min_confianza,
        )
        self.face=vision.FaceLandmarker.create_from_options(fo)
        self.pose=vision.PoseLandmarker.create_from_options(po)

    def cerrar(self):
        try: self.face.close()
        except: pass
        try: self.pose.close()
        except: pass

# ══════════════════════════════════════════════════════════════
# FUNCIONES DE MÉTRICAS
# Calculan cada parámetro a partir de los landmarks del rostro.
# ══════════════════════════════════════════════════════════════
def dist(a,b):
    # Distancia euclídea entre dos landmarks normalizados
    return np.sqrt((a.x-b.x)**2+(a.y-b.y)**2)

def blink_score(fr):
    # Score de cierre de ojos usando blendshapes de MediaPipe (0=abierto, 1=cerrado)
    try:
        bl={b.category_name:b.score for b in fr.face_blendshapes[0]}
        return (bl.get("eyeBlinkLeft",0)+bl.get("eyeBlinkRight",0))/2
    except: return 0.0

def apertura_boca(lm):
    # MAR: ratio entre distancia vertical y horizontal de la boca
    return dist(lm[BOCA_SUP],lm[BOCA_INF])/(dist(lm[BOCA_IZQ],lm[BOCA_DER])+1e-6)

def angulos_cabeza(lm):
    # Roll: inclinación lateral. Pitch: caída hacia delante.
    f,m,n=lm[FRENTE],lm[MENTON],lm[NARIZ_TIP]
    roll=abs(np.degrees(np.arctan2(m.x-f.x,m.y-f.y+1e-6)))
    eje_y=(f.y+m.y)/2
    pitch=abs((n.y-eje_y)/(abs(m.y-f.y)+1e-6))*90
    return roll,pitch

def detectar_caida(pr):
    # Mide si la nariz ha bajado respecto a los hombros y si hay asimetría postural
    try:
        lm=pr.pose_landmarks[0]; nariz=lm[POSE_NARIZ]
        yh=(lm[POSE_HOMBRO_IZQ].y+lm[POSE_HOMBRO_DER].y)/2
        return max(max(0.0,nariz.y-yh), abs(lm[POSE_HOMBRO_IZQ].y-lm[POSE_HOMBRO_DER].y)*0.6)
    except: return 0.0

def detectar_malestar(fr):
    # Combina 11 blendshapes de gestos involuntarios de dolor o malestar
    if not fr.face_blendshapes: return 0.0
    bl={b.category_name:b.score for b in fr.face_blendshapes[0]}
    return float(np.clip(sum(bl.get(n,0)*p for n,p in BLENDSHAPES_MALESTAR.items())/_PESO_MALESTAR,0,1))

# ══════════════════════════════════════════════════════════════
# DETECTOR PRINCIPAL DE LIPOTIMIA
# Combina todas las señales en un sistema de puntuación.
# Mantiene un historial de los últimos N niveles para suavizar
# cambios bruscos y evitar falsos positivos momentáneos.
# ══════════════════════════════════════════════════════════════
class DetectorLipotimia:
    def __init__(self, cfg):
        self.cfg=cfg
        # Filtro EMA individual para cada métrica
        self.f={k:EMA(cfg.alpha_ema) for k in ["blink","boca","roll","pitch","caida","palidez","malestar"]}
        self.frames_ojos=0          # contador de frames con ojos cerrados
        self.hist=deque(maxlen=cfg.ventana_riesgo)  # historial de niveles recientes

    def filtrar(self, blink,boca,roll,pitch,caida,palidez,malestar):
        # Aplica suavizado EMA a todas las métricas
        return tuple(self.f[k].update(v) for k,v in zip(
            ["blink","boca","roll","pitch","caida","palidez","malestar"],
            [blink,boca,roll,pitch,caida,palidez,malestar]))

    def evaluar(self, blink,boca,palidez,malestar,alerta_mirada,roll_ok,pitch_ok,caida_ok):
        c=self.cfg; senales=[]

        # Acumular señal de ojos cerrados solo si se mantiene suficientes frames
        self.frames_ojos=self.frames_ojos+1 if blink>c.umbral_blink else max(0,self.frames_ojos-1)
        if self.frames_ojos>c.frames_ojos_umbral: senales.append(("OJOS CERRADOS",3))

        # Señales faciales directas
        if boca>c.umbral_boca:        senales.append(("BOCA ABIERTA",1))
        if palidez>c.umbral_palidez:  senales.append(("PALIDEZ",2))
        if malestar>c.umbral_malestar: senales.append(("MALESTAR",1))
        if alerta_mirada:              senales.append(("MIRADA FIJA",3))

        # Señales posturales: solo si pasaron los filtros de velocidad y duración
        if roll_ok:  senales.append(("CABEZA LATERAL",2))
        if pitch_ok: senales.append(("CABEZA CAIDA",2))
        if caida_ok: senales.append(("CAIDA POSTURAL",3))

        # Coherencia: una señal aislada no sube de nivel (puede ser hablar, toser...)
        if len(senales)>=c.min_params_coherencia:
            puntos=sum(p for _,p in senales); alertas=[n for n,_ in senales]
        elif senales:
            # Señal aislada: se anota con asterisco pero no suma puntos
            puntos=0; alertas=[n+"*" for n,_ in senales]
        else:
            puntos=0; alertas=[]

        # Nivel: media de los últimos N frames para suavizar
        self.hist.append(0 if puntos==0 else 1 if puntos<=2 else 2)
        return int(round(np.mean(self.hist))), alertas

# ══════════════════════════════════════════════════════════════
# LOGGER
# Acumula todos los datos de la sesión en memoria.
# Al terminar genera el CSV con todos los frames y el PDF
# con el resumen clínico de la sesión.
# ══════════════════════════════════════════════════════════════
class Logger:
    NOMBRES={0:"NORMAL",1:"PRECAUCION",2:"ALERTA"}
    COLORES={"NORMAL":colors.HexColor("#1a7a3a"),"PRECAUCION":colors.HexColor("#0077aa"),"ALERTA":colors.HexColor("#cc2200")}

    def __init__(self):
        self._r=[]    # registro frame a frame
        self._ev=[]   # eventos de cambio de nivel (para línea de tiempo)
        self._np=-1   # nivel previo para detectar cambios
        self._ti=time.strftime("%H:%M:%S"); self._fecha=time.strftime("%d/%m/%Y")

    def escribir(self, e, blink,boca,roll,pitch,caida,palidez,malestar,ear,mov_iris,bostezo,total_b,nivel,alertas,fps):
        ts=time.strftime("%H:%M:%S")
        self._r.append({"ts":ts,"blink":round(blink,3),"boca":round(boca,3),"roll":round(roll,1),
            "pitch":round(pitch,1),"caida":round(caida,3),"palidez":round(palidez,3),
            "malestar":round(malestar,3),"ear":round(ear,3),"mov_iris":round(mov_iris,4),
            "nivel":nivel,"nivel_txt":self.NOMBRES[nivel],"alertas":";".join(alertas),
            "bostezo":1 if bostezo else 0,"total_bostezos":total_b,"fps":round(fps,1)})
        if nivel!=self._np:
            self._ev.append({"ts":ts,"nivel":self.NOMBRES[nivel],"alertas":";".join(alertas) or "—"})
            self._np=nivel

    def guardar_csv(self, ruta="historico_sesion.csv"):
        if not self._r: print("Sin datos"); return
        with open(ruta,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(self._r[0].keys()))
            w.writeheader(); w.writerows(self._r)
        print(f"✔ CSV → {ruta}")

    def guardar_pdf(self, ruta="informe_sesion.pdf", total_b=0, ultimo_b="—"):
        if not self._r: return
        doc=SimpleDocTemplate(ruta,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
        st=getSampleStyleSheet()
        et=ParagraphStyle("t",parent=st["Title"],fontSize=18,textColor=colors.HexColor("#1a1a2e"))
        es=ParagraphStyle("s",parent=st["Normal"],fontSize=10,textColor=colors.grey,spaceAfter=12)
        eh=ParagraphStyle("h",parent=st["Heading2"],fontSize=12,spaceBefore=14,spaceAfter=6,textColor=colors.HexColor("#1a1a2e"))
        en=ParagraphStyle("n",parent=st["Normal"],fontSize=9,leading=13)
        ee=ParagraphStyle("e",parent=st["Normal"],fontSize=8,leading=12,leftIndent=8)
        tf=self._r[-1]["ts"]; nf=len(self._r)
        niv=[r["nivel"] for r in self._r]
        nn,np_,na=niv.count(0),niv.count(1),niv.count(2)
        from collections import Counter
        ta=[a for r in self._r for a in r["alertas"].split(";") if a and a!="—"]
        top=Counter(ta).most_common(5)
        def med(c): v=[r[c] for r in self._r]; return round(sum(v)/len(v),3) if v else 0
        story=[]
        story.append(Paragraph("Informe de sesión — FainWatch",et))
        story.append(Paragraph(f"Fecha: {self._fecha} · Inicio: {self._ti} · Fin: {tf} · Frames: {nf}",es))
        story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor("#cccccc"),spaceAfter=10))
        story.append(Paragraph("Resumen de estados",eh))
        de=[["Estado","Frames","% tiempo"],
            ["NORMAL",str(nn),f"{100*nn/nf:.1f}%"],
            ["PRECAUCIÓN",str(np_),f"{100*np_/nf:.1f}%"],
            ["ALERTA",str(na),f"{100*na/nf:.1f}%"]]
        t=Table(de,colWidths=[5*cm,4*cm,4*cm])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),9),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f5f5f5"),colors.white]),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cccccc")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("TEXTCOLOR",(0,1),(-1,1),colors.HexColor("#1a7a3a")),
            ("TEXTCOLOR",(0,2),(-1,2),colors.HexColor("#0077aa")),
            ("TEXTCOLOR",(0,3),(-1,3),colors.HexColor("#cc2200")),
            ("FONTNAME",(0,1),(-1,-1),"Helvetica-Bold")]))
        story.append(t); story.append(Spacer(1,10))
        story.append(Paragraph("Alertas más frecuentes",eh))
        if top:
            da=[["Alerta","Veces"]]+[[a,str(c)] for a,c in top]
            ta2=Table(da,colWidths=[9*cm,4*cm])
            ta2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),9),
                ("ALIGN",(1,0),(1,-1),"CENTER"),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f5f5f5"),colors.white]),
                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cccccc")),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            story.append(ta2)
        else: story.append(Paragraph("Ninguna alerta.",en))
        story.append(Spacer(1,10))
        story.append(Paragraph("Bostezos",eh))
        story.append(Paragraph(f"Total: <b>{total_b}</b> · Último: <b>{ultimo_b}</b>",en))
        story.append(Spacer(1,10))
        story.append(Paragraph("Métricas medias",eh))
        dm=[["Métrica","Media"],["Blink",str(med("blink"))],["Boca",str(med("boca"))],
            ["Roll (°)",str(med("roll"))],["Pitch (°)",str(med("pitch"))],
            ["Caída",str(med("caida"))],["Palidez",str(med("palidez"))],
            ["Malestar",str(med("malestar"))],["EAR",str(med("ear"))]]
        tm=Table(dm,colWidths=[7*cm,6*cm])
        tm.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),9),
            ("ALIGN",(1,0),(1,-1),"CENTER"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f5f5f5"),colors.white]),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cccccc")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(tm); story.append(Spacer(1,10))
        story.append(Paragraph("Línea de tiempo",eh))
        for ev in self._ev:
            col=self.COLORES.get(ev["nivel"],colors.black)
            story.append(Paragraph(
                f'<font color="{col.hexval() if hasattr(col,"hexval") else "#333"}"><b>{ev["ts"]}</b></font>'
                f' → <b>{ev["nivel"]}</b> {ev["alertas"]}',ee))
        story.append(Spacer(1,14))
        story.append(HRFlowable(width="100%",thickness=0.5,color=colors.grey))
        story.append(Paragraph(f"FainWatch · {self._fecha}",
            ParagraphStyle("p",parent=st["Normal"],fontSize=7,textColor=colors.grey,alignment=TA_CENTER)))
        doc.build(story)
        print(f"✔ PDF → {ruta}")

# ══════════════════════════════════════════════════════════════
# VISUALIZACIÓN EN PANTALLA
# Dibuja sobre el frame de la cámara el panel de métricas,
# los indicadores de alertas y la barra de nivel de riesgo.
# ══════════════════════════════════════════════════════════════
def _barra(frame, x, y, w_max, valor, maximo, color):
    # Dibuja una barra de progreso que representa visualmente el valor de cada métrica
    ancho=int(min(valor/max(maximo,1e-6),1.0)*w_max)
    cv2.rectangle(frame,(x,y),(x+w_max,y+7),(30,30,40),-1)
    if ancho>0: cv2.rectangle(frame,(x,y),(x+ancho,y+7),color,-1)

def dibujar(frame, vals, calibrado, frames_cal, total_cal, bl=None, mov_rapido=False):
    blink,boca,roll,pitch,caida,palidez,malestar,ear,mov_iris,bostezo,total_b,nivel,alertas,fps = vals
    h,w=frame.shape[:2]

    # Cabecera: nombre, indicador de grabación y fps
    cv2.rectangle(frame,(0,0),(w,48),(8,8,18),-1)
    cv2.putText(frame,"FainWatch",(14,32),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,100),2)
    cv2.circle(frame,(155,24),7,(0,60,220),-1)
    cv2.putText(frame,"REC",(168,29),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,80,255),1)
    cv2.putText(frame,f"{fps:.0f}fps",(w-65,29),cv2.FONT_HERSHEY_SIMPLEX,0.42,(70,70,80),1)
    cv2.putText(frame,"Q = parar",(w-130,29),cv2.FONT_HERSHEY_SIMPLEX,0.35,(50,50,60),1)

    # Barra de calibración: muestra el progreso de aprendizaje del donante
    if not calibrado:
        p=int((frames_cal/max(total_cal,1))*(w-20))
        cv2.rectangle(frame,(10,h-32),(w-10,h-16),(30,30,40),-1)
        cv2.rectangle(frame,(10,h-32),(10+p,h-16),(0,200,100),-1)
        cv2.putText(frame,f"Calibrando... {frames_cal}/{total_cal}",(14,h-38),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,200,150),1)
        return

    # Panel de métricas con barras de progreso (esquina derecha)
    px=w-148
    cv2.rectangle(frame,(px-6,52),(w-4,370),(8,8,18),-1)
    cv2.rectangle(frame,(px-6,52),(w-4,370),(0,80,40),1)
    cv2.putText(frame,"METRICAS",(px,70),cv2.FONT_HERSHEY_SIMPLEX,0.34,(0,180,80),1)
    cv2.line(frame,(px,74),(w-8,74),(0,60,30),1)

    V=(0,200,80); A=(0,200,200); G=(80,80,90); bw=90
    mets=[
        ("OJOS",   blink,    1.0,  V),   # score de cierre de ojos
        ("PALIDEZ",palidez,  1.0,  A),   # desviación de color respecto al baseline
        ("MALESTAR",malestar,1.0,  A),   # expresión de malestar
        ("ROLL",   roll,     45.0, V),   # inclinación lateral de cabeza
        ("PITCH",  pitch,    45.0, V),   # caída hacia delante de cabeza
        ("EAR",    ear,      0.5,  V),   # apertura ocular actual
        ("IRIS",   mov_iris, 0.05, G),   # movimiento del iris
        ("CAIDA",  caida,    0.5,  A),   # caída postural (nariz vs hombros)
    ]
    for i,(lb,val,mx,col) in enumerate(mets):
        yr=88+i*34
        cv2.putText(frame,lb,(px,yr),cv2.FONT_HERSHEY_SIMPLEX,0.33,(110,110,120),1)
        _barra(frame,px,yr+3,bw,val,mx,col)
        vs=f"{val:.2f}" if mx==1.0 else f"{val:.1f}"
        cv2.putText(frame,vs,(px+bw+4,yr+10),cv2.FONT_HERSHEY_SIMPLEX,0.32,(140,180,160),1)
    cv2.line(frame,(px,362),(w-8,362),(0,50,25),1)
    cv2.putText(frame,f"Bostezos:{total_b}",(px,378),cv2.FONT_HERSHEY_SIMPLEX,0.34,(100,100,110),1)

    # Indicador de movimiento voluntario ignorado
    if mov_rapido:
        cv2.rectangle(frame,(10,52),(225,76),(18,18,8),-1)
        cv2.rectangle(frame,(10,52),(225,76),(30,120,170),1)
        cv2.putText(frame,"MOV VOLUNTARIO IGNORADO",(14,69),cv2.FONT_HERSHEY_SIMPLEX,0.37,(40,160,200),1)

    # Indicador de bostezo en curso
    if bostezo:
        cv2.rectangle(frame,(10,82),(170,105),(18,18,8),-1)
        cv2.putText(frame,"BOSTEZO",(14,100),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,220,255),2)

    # Tarjetas de alertas activas (esquina izquierda)
    # Rojo = señal que suma puntos | Amarillo = señal aislada (observación)
    al_limpias=[a for a in alertas if not a.endswith("*")]
    al_aisladas=[a.rstrip("*") for a in alertas if a.endswith("*")]
    yc=118
    for a in al_limpias[:3]:
        cv2.rectangle(frame,(10,yc),(190,yc+24),(35,15,15),-1)
        cv2.rectangle(frame,(10,yc),(190,yc+24),(80,30,30),1)
        cv2.putText(frame,a,(14,yc+16),cv2.FONT_HERSHEY_SIMPLEX,0.40,(220,100,100),1)
        yc+=28
    for a in al_aisladas[:2]:
        cv2.rectangle(frame,(10,yc),(190,yc+24),(28,28,15),-1)
        cv2.rectangle(frame,(10,yc),(190,yc+24),(70,70,25),1)
        cv2.putText(frame,f"{a} (obs)",(14,yc+16),cv2.FONT_HERSHEY_SIMPLEX,0.36,(160,160,60),1)
        yc+=28

    # Barra inferior de nivel de riesgo: verde, amarillo o rojo
    COL={0:((0,35,0),(0,255,0)),1:((35,35,0),(0,255,255)),2:((35,0,0),(0,0,255))}
    bg,ct=COL[nivel]
    NOM={0:"NORMAL",1:"PRECAUCION",2:"ALERTA"}
    cv2.rectangle(frame,(0,h-72),(w,h),bg,-1)
    cv2.rectangle(frame,(0,h-72),(w,h-69),ct,-1)
    cv2.putText(frame,NOM[nivel],(18,h-38),cv2.FONT_HERSHEY_SIMPLEX,1.05,ct,2)
    ok=["Ojos OK","Postura OK","Piel OK","Sin malestar"] if not al_limpias else al_limpias
    cv2.putText(frame,"  |  ".join(ok),(210,h-38),cv2.FONT_HERSHEY_SIMPLEX,0.38,ct,1)

    # Pie: valores de calibración del donante actual
    if bl and bl.listo:
        cv2.putText(frame,
            f"Calibrado  EAR={bl.ear_media:.3f}  V={bl.ref_V.get(10,0):.0f}  S={bl.ref_S.get(10,0):.0f}",
            (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.28,(40,55,45),1)

# ══════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL
# Abre la cámara directamente con OpenCV (sin navegador ni red),
# procesa cada frame y muestra el resultado en tiempo real.
# Pulsa Q para parar y generar el informe.
# ══════════════════════════════════════════════════════════════
def ejecutar(cfg=None):
    if cfg is None: cfg=Config()

    bl   = BaselineColor()          # aprende el color de piel del donante
    mods = ModelosMP(cfg)           # carga los modelos de MediaPipe
    det  = DetectorLipotimia(cfg)   # evalúa el nivel de riesgo
    mir  = DetectorMiradaFija(cfg,bl)  # detecta mirada fija
    bos  = DetectorBostezo(cfg)     # detecta bostezos
    fmv  = FiltroMovimientoVoluntario(cfg)  # filtra movimientos voluntarios
    cal  = Calibrador(cfg,bl)       # calibra umbrales al donante
    log  = Logger()                 # acumula datos para CSV y PDF

    # Apertura directa de la cámara — sin JS, sin red, sin latencia
    cap=cv2.VideoCapture(cfg.camara_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,30)

    if not cap.isOpened():
        print("No se puede abrir la cámara. Cambia camara_idx en Config.")
        return

    print("✔ Cámara abierta — pulsa Q para parar y generar el informe")
    cv2.namedWindow("FainWatch", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FainWatch", 800, 600)

    t_prev=time.time(); n_frame=0; caida=0.0

    try:
        while True:
            ret,frame=cap.read()
            if not ret: print("Error leyendo cámara"); break

            n_frame+=1
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            ts=int(time.time()*1000)

            fr=mods.face.detect_for_video(mp_img,ts)
            # Pose cada 3 frames: es el modelo más lento y la caída postural
            # es un proceso lento que no necesita actualizarse en cada frame
            if n_frame%3==0:
                pr=mods.pose.detect_for_video(mp_img,ts)
                if pr.pose_landmarks: caida=detectar_caida(pr)

            blink=boca=roll=pitch=palidez=malestar=ear=mov_iris=0.0
            alerta_mir=bostezo=False; lm=None

            if fr.face_landmarks:
                lm=fr.face_landmarks[0]
                blink=blink_score(fr)
                boca=apertura_boca(lm)
                roll,pitch=angulos_cabeza(lm)
                malestar=detectar_malestar(fr)
                palidez=bl.score_palidez(frame,lm)
                ear,mov_iris,alerta_mir=mir.actualizar(lm)
                bostezo=bos.actualizar(lm,fr)

            cal.agregar(frame,lm,blink,boca,roll,pitch,caida,malestar,ear=ear)
            blink,boca,roll,pitch,_,palidez,malestar=det.filtrar(blink,boca,roll,pitch,caida,palidez,malestar)
            ro,pi,ca,rapido=fmv.actualizar(roll,pitch,caida)
            nivel,alertas=det.evaluar(blink,boca,palidez,malestar,alerta_mir,ro,pi,ca)

            now=time.time(); fps=1.0/max(now-t_prev,1e-6); t_prev=now
            log.escribir(None,blink,boca,roll,pitch,caida,palidez,malestar,ear,mov_iris,
                         bostezo,bos.total,nivel,alertas,fps)

            vals=(blink,boca,roll,pitch,caida,palidez,malestar,ear,mov_iris,
                  bostezo,bos.total,nivel,alertas,fps)
            dibujar(frame,vals,cal.calibrado,len(cal.buf["blink"]),cfg.frames_calibracion,bl,rapido)
            cv2.imshow("FainWatch",frame)

            if cv2.waitKey(1)&0xFF==ord('q'): break

    finally:
        cap.release(); cv2.destroyAllWindows(); mods.cerrar()
        print(f"\n✔ Sesión terminada — {len(log._r)} frames | {bos.total} bostezos")
        log.guardar_csv()
        log.guardar_pdf(total_b=bos.total,ultimo_b=bos.ultimo)

# ══════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# Modifica aquí los parámetros antes de ejecutar.
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ejecutar(Config(
        camara_idx=1,              # 0=cámara por defecto, 1=segunda cámara...
        frames_calibracion=50,     # frames iniciales de calibración (~4s a 12fps)
        umbral_palidez=0.25,       # sensibilidad a la palidez (bajar = más sensible)
        umbral_malestar=0.30,      # sensibilidad a expresión de malestar
        ear_margen_sigma=2.0,      # desviaciones para mirada fija (subir = menos sensible)
        umbral_movimiento_iris=0.008,
        frames_mirada_fija=22,
        umbral_mar_bostezo=0.45,
        umbral_squint_bostezo=0.25,
        frames_bostezo=14,
        umbral_velocidad_giro=3.5,
        min_params_coherencia=2,
        frames_postural_minimo=8,
    ))
