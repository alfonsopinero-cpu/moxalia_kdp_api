# main.py
import os
import json
import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, QSize
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QMessageBox,
    QPlainTextEdit,
    QAbstractItemView,
    QFrame,
    QProgressBar,
    QFileDialog,
    QInputDialog,  
    

)
from PySide6.QtGui import QIcon, QPixmap, QColor, QBrush, QCursor, QDesktopServices, QAction
from PySide6.QtCore import QUrl

# =========================
# CONFIG
# =========================
load_dotenv()

IMAGE_RESOLUTIONS = {
    "Testing (1024 x 1024)": "1024x1024",
    "Working (1024 x 1536)": "1024x1536",
    "Final KDP (2550 x 3300)": "2550x3300",
}
REQ_IMAGENES_PATH = Path.cwd() / "requerimientos_imagenes.txt"
REQ_TEXTO_PATH = Path.cwd() / "requerimientos_texto.txt"

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")

OUT_DIR = Path.cwd() / "illustraciones"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = Path.cwd() / "book_state.json"

PROJECTS_DIR = Path.cwd() / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# DATA MODELS
# =========================
class BookSpec(BaseModel):
    book_id: str = Field(..., min_length=1)
    publico: str
    tema: str
    alcance_geografico: str
    tipo_imagenes: str
    numero_imagenes: int = Field(..., ge=1, le=500)
    tamano_libro: str
    aspect_ratio: str
    dificultad_inicial: str
    dificultad_final: str
    idioma_texto: str


class IllustrationRow(BaseModel):
    ID: str
    Categoria: str
    Publico: str
    Pais_Region: str
    Monumento: str
    Main_foco: str

    Difficulty_block: str  # Bajo | Medio | Alto | Extremo
    Difficulty_D: int
    Line_thickness_L: float
    White_space_W: float
    Complexity_C: float

    Nombre_fichero_editorial: str
    Nombre_fichero_descargado: str
    Comentario_editorial: str
    Prompt_core: str
    Prompt_final: str
    Prompt_negativo: str
    approval_status: str = "pending"  # pending | approved | rejected
    rejection_reason: str = ""

def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _safe_slug(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9-_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "book"

def load_requirement_file(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "content": ""}

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {"status": "empty", "content": ""}

    return {"status": "ok", "content": content}

def build_prompt_final(
    *,
    spec: BookSpec,
    row: IllustrationRow,
    req_img: str,
    req_txt: str,
) -> str:
    return (
        "=== COLORING BOOK PROMPT ===\n\n"
        "BOOK CONTEXT\n"
        f"Title / Theme: {spec.tema}\n"
        f"Audience: {spec.publico}\n"
        f"Geographic scope: {spec.alcance_geografico}\n"
        f"Language: {spec.idioma_texto}\n\n"
        "EDITORIAL IMAGE REQUIREMENTS\n"
        f"{req_img}\n\n"
        "EDITORIAL TEXT REQUIREMENTS\n"
        f"{req_txt}\n\n"
        "DIFFICULTY PARAMETERS (MANDATORY)\n"
        f"Difficulty level: {row.Difficulty_D}/100\n"
        f"Block: {row.Difficulty_block}\n"
        f"Line thickness (L): {row.Line_thickness_L}\n"
        f"White space dominance (W): {row.White_space_W}\n"
        f"Structural complexity (C): {row.Complexity_C}\n"
        "DIFFICULTY INTERPRETATION (MANDATORY)\n"
        "Higher difficulty means MORE distinct drawable elements.\n"
        "Increase the number of architectural sub-elements, foreground elements, and mid-ground details with difficulty.\n"
        "Difficulty must be visible at first glance when comparing pages.\n"
        "SCENE DESCRIPTION\n"
        f"{row.Prompt_core}"
    ).strip()

# =========================
# OPENAI HELPERS
# =========================
def _extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{\s*\".*?\}\s*)", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Could not extract JSON from model output")

DIFFICULTY_RANGES = {
    "Bajo": (0, 25),
    "Medio": (26, 50),
    "Alto": (51, 75),
    "Extremo": (76, 100),
}

def build_difficulty_sequence(total: int, start: str, end: str) -> list[dict]:
    if start not in DIFFICULTY_RANGES or end not in DIFFICULTY_RANGES:
        raise ValueError("Invalid difficulty block")

    start_min, _ = DIFFICULTY_RANGES[start]
    _, end_max = DIFFICULTY_RANGES[end]

    if start_min > end_max:
        raise ValueError("Initial difficulty higher than final difficulty")

    if total == 1:
        values = [start_min]
    else:
        values = [
            round(start_min + i * (end_max - start_min) / (total - 1))
            for i in range(total)
        ]

    sequence = []
    for d in values:
        block = next(
            b for b, (mn, mx) in DIFFICULTY_RANGES.items() if mn <= d <= mx
        )

        x = d / 100.0
        sequence.append({
            "D": d,
            "block": block,
            "L": round(0.25 + 0.55 * x, 2),
            "W": round(0.75 - 0.55 * x, 2),
            "C": round(0.15 + 0.75 * x, 2),
        })

    return sequence
    

def generate_illustrations(
    client: OpenAI,
    spec: BookSpec,
    text_model: str,
    *,
    req_img: str,
    req_txt: str,
    progress_cb=None,
) -> List[IllustrationRow]:

    difficulty_sequence = build_difficulty_sequence(
        total=spec.numero_imagenes,
        start=spec.dificultad_inicial,
        end=spec.dificultad_final,
    )

    if progress_cb:
        progress_cb(15, "Difficulty sequence built")
    system = (
        "You are a professional editorial engine for Amazon KDP coloring books.\n"
        "Return ONLY valid JSON (no markdown, no commentary).\n"
        "You must output a JSON array of objects with EXACT keys:\n"
        "ID, Categoria, Publico, Pais_Region, Monumento, Main_foco, "
        "Nombre_fichero_editorial, Nombre_fichero_descargado, "
        "Comentario_editorial, Prompt_core, Prompt_negativo\n"
        "Prompt_core MUST contain ONLY the scene description.\n"
        "DO NOT include difficulty, requirements, formatting rules, or meta instructions.\n"
        "Main_foco MUST describe: the main monument, the real, plausible surrounding environment, a human-scale point of view\n"
        "Main_foco must describe a REALISTIC and geographically plausible scene. Do NOT invent landscape elements that do not exist at the real location.\n"
        "Comentario_editorial MUST be a poetic and inspirational text.\n"
        "It MUST be between 5 and 10 lines long.\n"
        "Each line should be a full, meaningful sentence.\n"
        "The text must describe an emotional or reflective moment connected to the scene.\n"
        "It must NOT describe technical, historical, or architectural details.\n"
        "It must NOT explain the monument.\n"
        "Constraints:\n"
        "All format, aspect ratio, audience and theme values come from book_spec.\n"
        "Do NOT infer or override these values\n"
        f"Exactly {spec.numero_imagenes} rows.\n"
        "ID must be a zero-padded integer string like 001, 002, ...\n"
        "Do not repeat the same country in consecutive rows.\n"
        "The order of illustrations is final.\n"
        "Prompts MUST be black-and-white line art for coloring books.\n"
        "No shading, no gray, no fills.\n"
    )
    user = {
        "book_spec": spec.model_dump(),
    }

    if progress_cb:
        progress_cb(30, "Calling OpenAI (text model)")

    r = client.chat.completions.create(
        model=text_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.3,
    )

    raw = r.choices[0].message.content
    if progress_cb:
        progress_cb(60, "Parsing model response")

    data = _extract_json(raw)

    if not isinstance(data, list):
        raise ValueError("Model did not return a JSON array")

    rows: List[IllustrationRow] = []

    for i, obj in enumerate(data):
        d = difficulty_sequence[i]

        row = IllustrationRow(
            **obj,
            Prompt_final="",  # placeholder
            Difficulty_D=d["D"],
            Difficulty_block=d["block"],
            Line_thickness_L=d["L"],
            White_space_W=d["W"],
            Complexity_C=d["C"],
        )

        row.Prompt_final = build_prompt_final(
            spec=spec,
            row=row,
            req_img=req_img,
            req_txt=req_txt,
        )
        rows.append(row)

    if len(rows) != spec.numero_imagenes:
        raise ValueError(
            f"Row count mismatch: expected {spec.numero_imagenes}, got {len(rows)}"
        )
    if progress_cb:
        progress_cb(85, "Validating results")

    # VALIDACIÓN DURA — PAÍSES NO CONSECUTIVOS
    for i in range(1, len(rows)):
        if rows[i].Pais_Region == rows[i - 1].Pais_Region:
            raise ValueError(
                f"Country repetition at index {i}: {rows[i].Pais_Region}"
            )
    if progress_cb:
        progress_cb(100, "Done")

    return rows


def generate_image_png(
    client: OpenAI,
    prompt: str,
    negative: str,
    out_path: Path,
    image_model: str,
    image_resolution: str,
    signals=None,
    progress_cb=None,
) -> None:
    final_prompt = (
        "BLACK AND WHITE COLORING BOOK LINE ART.\n"
        "Only clean black outlines on white background.\n"
        "No grayscale, no shading, no hatching, no fills.\n"
        "High contrast, printable, crisp lines.\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"NEGATIVE:\n{negative}\n"
    )
    if progress_cb:
        progress_cb(0, "Starting image generation")
    img = client.images.generate(
        model=image_model,
        prompt=final_prompt,
        size=image_resolution,
    )
    if progress_cb:
        progress_cb(70, "Image generated, saving file")
    b64 = img.data[0].b64_json
    out_path.write_bytes(base64.b64decode(b64))
    if progress_cb:
        progress_cb(100, "Image ready")

# =========================
# STATE I/O
# =========================
def save_state(spec: BookSpec, rows: List[IllustrationRow]) -> None:
    payload = {
        "spec": spec.model_dump(),
        "rows": [r.model_dump() for r in rows],
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state() -> Optional[Dict[str, Any]]:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


# =========================
# WORKERS
# =========================
class WorkerSignals(QObject):
    ok = Signal(object)
    progress = Signal(int, str)  # porcentaje, mensaje
    err = Signal(str)


class GenerateBookWorker(QRunnable):
    def __init__(self, client, spec, text_model, *, req_img: str, req_txt: str):
        super().__init__()
        self.client = client
        self.spec = spec
        self.text_model = text_model
        self.req_img = req_img
        self.req_txt = req_txt
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.progress.emit(10, "Starting...")
            rows = generate_illustrations(
                self.client,
                self.spec,
                self.text_model,
                req_img=self.req_img,
                req_txt=self.req_txt,
                progress_cb=lambda p, m: self.signals.progress.emit(p, m),
            )

            self.signals.ok.emit(rows)
        except Exception as e:
            self.signals.err.emit(str(e))




class GenerateImageWorker(QRunnable):
    def __init__(
        self,
        client: OpenAI,
        row: IllustrationRow,
        image_model: str,
        image_resolution: str,
        images_dir: Path,
    ):
        super().__init__()
        self.client = client
        self.row = row
        self.image_model = image_model
        self.image_resolution = image_resolution
        self.images_dir = images_dir
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.images_dir / f"{self.row.ID}.png"
            generate_image_png(
                self.client,
                prompt=self.row.Prompt_final,
                negative=self.row.Prompt_negativo,
                out_path=out_path,
                image_model=self.image_model,
                image_resolution=self.image_resolution,
                progress_cb=lambda p, m: self.signals.progress.emit(p, m),
            )
            self.signals.ok.emit(str(out_path))
        except Exception as e:
            self.signals.err.emit(str(e))


# =========================
# UI
# =========================

class BlockingOverlay(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self._lock_label = False
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(0,0,0,160);")
        self.setGeometry(parent.rect())
        self.setVisible(False)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)


        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Working...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            color: white;
            font-size: 20px;
            letter-spacing: 0.5px;
            font-weight: bold;
        """)
        layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setFixedHeight(23)
        self.bar.setStyleSheet("""
        QProgressBar {
            border: 1px solid #444;
            border-radius: 6px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 6px;
        }
        """)
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)
    def start_indeterminate(self, text="🤖 AI is thinking…"):
        self._lock_label = True
        self.label.setText(text)
        self.bar.setRange(0, 0)   # barra animada
        self.setVisible(True)
        self.raise_()
    def start(self, text="Working..."):
        self._lock_label = False
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.label.setText(text)
        self.setVisible(True)
        self.raise_()   # 🔴 CLAVE

    def update(self, value: int, text: str):
        if self.bar.minimum() == 0 and self.bar.maximum() == 0:
            # transición controlada a determinista
            self.bar.setRange(0, 100)
            self._lock_label = False

        self.bar.setValue(value)

        if not self._lock_label:
            self.label.setText(text)
    def stop(self):
        self._lock_label = False
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self.setVisible(False)

class ImageReviewDialog(QDialog):
    def __init__(self, parent, row: IllustrationRow, image_path: Path):
        super().__init__(parent)          # ✅ ESTA LÍNEA ES LA CLAVE
        self.row = row                   # ✅ para _approve/_reject

        self.setWindowTitle(f"Review image {row.ID}")
        self.setModal(True)


        # Tamaño mínimo pensado para 14" (1920x1200 @150%)
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)   # tamaño inicial razonable
        self.setSizeGripEnabled(True)

        root = QHBoxLayout(self)

        # =========================
        # LEFT: IMAGE (MAX SPACE)
        # =========================
        left = QVBoxLayout()
        root.addLayout(left, 3)  # 75% espacio

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: #111;")

        pix = QPixmap(str(image_path))
        self.img_label.setPixmap(
            pix.scaled(
                900, 1200,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        left.addWidget(self.img_label)

        # =========================
        # RIGHT: INFO PANEL
        # =========================
        right = QVBoxLayout()
        right.setSpacing(8)
        root.addLayout(right, 2)  # 25% espacio

        # --- Prompt ---
        prompt_box = QGroupBox("Prompt (final + negative)")
        prompt_layout = QVBoxLayout(prompt_box)

        txt_prompt = QPlainTextEdit()
        txt_prompt.setReadOnly(True)
        txt_prompt.setPlainText(
            row.Prompt_final + "\n\n--- NEGATIVE PROMPT ---\n\n" + row.Prompt_negativo
        )
        prompt_layout.addWidget(txt_prompt)
        right.addWidget(prompt_box, 2)

        # --- Editorial text ---
        text_box = QGroupBox("Inspirational text")
        text_layout = QVBoxLayout(text_box)

        self.txt_editorial = QPlainTextEdit()
        self.txt_editorial.setPlainText(row.Comentario_editorial)
        text_layout.addWidget(self.txt_editorial)
        right.addWidget(text_box, 1)

        # --- Rejection reason ---
        self.reject_reason = QLineEdit()
        self.reject_reason.setPlaceholderText("Reason for rejection (required)")
        if row.rejection_reason:
            self.reject_reason.setText(row.rejection_reason)
        right.addWidget(self.reject_reason)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_approve = QPushButton("Approve")
        self.btn_reject = QPushButton("Reject")
        self.btn_cancel = QPushButton("Close")

        btn_layout.addWidget(self.btn_approve)
        btn_layout.addWidget(self.btn_reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)

        right.addLayout(btn_layout)

        # --- Signals ---
        self.btn_approve.clicked.connect(self._approve)
        self.btn_reject.clicked.connect(self._reject)
        self.btn_cancel.clicked.connect(self.reject)


    def _approve(self):
        self.row.approval_status = "approved"
        self.row.rejection_reason = ""
        self.accept()

    def _reject(self):
        reason = self.reject_reason.text().strip()
        if not reason:
            QMessageBox.warning(
                self, "Missing reason", "Please provide a reason for rejection."
            )
            return
        self.row.approval_status = "rejected"
        self.row.rejection_reason = reason
        self.accept()

class MainWindow(QMainWindow):
    def get_selected_image_resolution(self) -> str:
        label = self.cb_image_resolution.currentText()
        return IMAGE_RESOLUTIONS[label]

    def _refresh_requirement(self, which: str):
        if which == "img":
            data = load_requirement_file(REQ_IMAGENES_PATH)
            self.req_imagenes = data
            self._set_req_label(
                self.lbl_req_img,
                "Image requirements",
                data["status"],
            )

        elif which == "txt":
            data = load_requirement_file(REQ_TEXTO_PATH)
            self.req_texto = data
            self._set_req_label(
                self.lbl_req_txt,
                "Text requirements",
                data["status"],
            )

    def on_create_new_book(self):   
        # Caso A: no hay proyecto cargado
        if self.project_dir is None:
            self.reset_project_state()
            return

        # Caso B: hay proyecto cargado
        reply = QMessageBox.question(
            self,
            "Save project?",
            "There is an open project.\nDo you want to save it before creating a new one?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        )

        if reply == QMessageBox.Cancel:
            return

        if reply == QMessageBox.Yes:
            self.on_save_project()
            # si sigue existiendo proyecto, algo falló → no continuar
            if self.project_dir is not None:
                self.reset_project_state()
            return

        # reply == No → segunda confirmación
        confirm = QMessageBox.question(
            self,
            "Discard changes?",
            "All unsaved changes will be lost.\nAre you sure?",
            QMessageBox.Yes | QMessageBox.Cancel,
        )

        if confirm == QMessageBox.Yes:
            self.reset_project_state()



    def _on_req_clicked(self, which: str):
        if which == "img":
            status = self.req_imagenes["status"]
            path = REQ_IMAGENES_PATH
        else:
            status = self.req_texto["status"]
            path = REQ_TEXTO_PATH

        if status == "ok":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self._refresh_requirement(which)

    def resizeEvent(self, event):
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(event)
    DISPLAY_COLS = [
        "Imagen",
        "ID",
        "Título",
        "Público",
        "País / Región",
        "Monumento",
        "Foco principal",
        "Bloque",
        "Nivel dificultad",
    ]
    TABLE_COLUMN_MAP = [
        ("ID", "ID"),
        ("Título", "Categoria"),
        ("Público", "Publico"),
        ("País / Región", "Pais_Region"),
        ("Monumento", "Monumento"),
        ("Foco principal", "Main_foco"),
        ("Bloque", "Difficulty_block"),
        ("Nivel dificultad", "Difficulty_D"),
    ]
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KDP Coloring Book Builder (MVP)")
        self.resize(1400, 900)              # tamaño inicial razonable
        self.setMinimumSize(1280, 800)      # 👈 clave para pantallas 14"


        self.client = OpenAI()
        self.pool = QThreadPool.globalInstance()
        self.overlay = BlockingOverlay(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.hide()
        self.req_imagenes = load_requirement_file(REQ_IMAGENES_PATH)
        self.req_texto = load_requirement_file(REQ_TEXTO_PATH)
        self.project_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None

        self.rows: List[IllustrationRow] = []
        self.current_spec: Optional[BookSpec] = None

        # New: Available model lists (can be extended at will)
        self.available_text_models = [
            DEFAULT_TEXT_MODEL,
            "gpt-4.1-mini",
            "gpt-4",
            "gpt-3.5-turbo-1106",
        ]
        self.available_image_models = [
            DEFAULT_IMAGE_MODEL,
            "dall-e-3",
            "gpt-image-1.5",
        ]

        # ----- MENU BAR -----
        menubar = self.menuBar()
        archivo_menu = menubar.addMenu("Archivo")
        acciones_menu = menubar.addMenu("Acciones")

        # Archivo actions
        self.action_crear_nuevo_libro = QAction("Crear nuevo libro", self)
        self.action_crear_nuevo_libro.triggered.connect(self.on_create_new_book)
        archivo_menu.addAction(self.action_crear_nuevo_libro)

        self.action_abrir_libro = QAction("Abrir libro existente", self)
        self.action_abrir_libro.triggered.connect(self.on_open_existing_book)
        archivo_menu.addAction(self.action_abrir_libro)

        self.action_guardar_proyecto = QAction("Guardar proyecto", self)
        self.action_guardar_proyecto.triggered.connect(self.on_save_project)
        archivo_menu.addAction(self.action_guardar_proyecto)

        # Acciones actions
        self.action_generar_libro = QAction("Generar libro", self)
        self.action_generar_libro.setEnabled(False)
        self.action_generar_libro.triggered.connect(self.on_generate_book)
        acciones_menu.addAction(self.action_generar_libro)

        self.action_generar_imagen = QAction("Generar imagen seleccionada", self)
        self.action_generar_imagen.setEnabled(False)
        self.action_generar_imagen.triggered.connect(self.on_generate_image)
        acciones_menu.addAction(self.action_generar_imagen)

        # ----- CENTRAL UI -----
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QVBoxLayout()
        main.addLayout(left, 3)

        project_box = QGroupBox("Project")
        left.addWidget(project_box)
        pvl = QVBoxLayout(project_box)

        self.lbl_project = QLabel("No book loaded")
        pvl.addWidget(self.lbl_project)

        pbtns = QHBoxLayout()
        pvl.addLayout(pbtns)

        self.btn_create_book = QPushButton("Create new book")
        self.btn_open_book = QPushButton("Open existing book")  # will implement later
        pbtns.addWidget(self.btn_create_book)
        pbtns.addWidget(self.btn_open_book)

        # Connect (keep connections for future code compatibility, but hide)
        self.btn_create_book.clicked.connect(self.on_create_new_book)
        self.btn_open_book.clicked.connect(self.on_open_existing_book)

        self.btn_create_book.hide()
        self.btn_open_book.hide()

        right = QVBoxLayout()
        main.addLayout(right, 2)
        
        form_box = QGroupBox("Book inputs")
        req_box = QGroupBox("Editorial requirements")
        left.addWidget(req_box)
        req_layout = QHBoxLayout(req_box)


        self.lbl_req_img = QLabel()
        self.lbl_req_txt = QLabel()

        self.lbl_req_img.setCursor(QCursor(Qt.PointingHandCursor))
        self.lbl_req_txt.setCursor(QCursor(Qt.PointingHandCursor))

        self.lbl_req_img.mousePressEvent = lambda e: self._on_req_clicked("img")
        self.lbl_req_txt.mousePressEvent = lambda e: self._on_req_clicked("txt")

        self._set_req_label(self.lbl_req_img, "Image requirements", self.req_imagenes["status"])
        self._set_req_label(self.lbl_req_txt, "Text requirements", self.req_texto["status"])

        self.lbl_req_img.setStyleSheet("text-decoration: underline;")
        self.lbl_req_txt.setStyleSheet("text-decoration: underline;")

        req_layout.addWidget(self.lbl_req_img)
        req_layout.addWidget(self.lbl_req_txt)

        left.addWidget(form_box)
        form_layout = QHBoxLayout(form_box)

        form_left = QFormLayout()
        form_right = QFormLayout()

        form_layout.addLayout(form_left)
        form_layout.addLayout(form_right)

        self.ed_book_id = QLineEdit()
        self.cb_publico = QComboBox()
        self.cb_publico.addItems(["Adultos", "Niños"])
        self.ed_tema = QLineEdit()
        self.ed_alcance = QLineEdit()
        self.ed_tipo = QLineEdit("Monumentos y paisajes")
        self.sp_num = QSpinBox()
        self.sp_num.setRange(1, 500)
        self.sp_num.setValue(50)

        self.ed_tamano = QLineEdit("8.5x11")
        self.ed_aspect = QLineEdit("4:5")

        self.cb_dif_ini = QComboBox()
        self.cb_dif_ini.addItems(["Bajo", "Medio", "Alto", "Extremo"])
        self.cb_dif_fin = QComboBox()
        self.cb_dif_fin.addItems(["Bajo", "Medio", "Alto", "Extremo"])
        self.cb_dif_fin.setCurrentText("Alto")

        self.cb_idioma = QComboBox()
        self.cb_idioma.addItems(["es", "en", "it", "fr", "de"])
        self.cb_idioma.setCurrentText("es")

        # Model selectors
        self.cb_text_model = QComboBox()
        for m in self.available_text_models:
            self.cb_text_model.addItem(m)
        # Use current as default
        idx_text = self.cb_text_model.findText(DEFAULT_TEXT_MODEL)
        if idx_text != -1:
            self.cb_text_model.setCurrentIndex(idx_text)
        form_right.addRow("Text model", self.cb_text_model)

        self.cb_image_resolution = QComboBox()
        self.cb_image_resolution.addItems(IMAGE_RESOLUTIONS.keys())
        self.cb_image_resolution.setCurrentText("Testing (1024 x 1024)")
        form_right.addRow("Image resolution", self.cb_image_resolution)

        self.cb_image_model = QComboBox()
        for m in self.available_image_models:
            self.cb_image_model.addItem(m)
        idx_image = self.cb_image_model.findText(DEFAULT_IMAGE_MODEL)
        if idx_image != -1:
            self.cb_image_model.setCurrentIndex(idx_image)
        form_right.addRow("Image model", self.cb_image_model)
        form_left.addRow("Book name", self.ed_book_id)
        form_left.addRow("Publico", self.cb_publico)
        form_left.addRow("Tema", self.ed_tema)
        form_left.addRow("Alcance geográfico", self.ed_alcance)
        form_left.addRow("Tipo de imágenes", self.ed_tipo)
        form_left.addRow("Número de imágenes", self.sp_num)
        form_left.addRow("Tamaño libro", self.ed_tamano)
        form_left.addRow("Aspect ratio", self.ed_aspect)
        form_right.addRow("Dificultad inicial", self.cb_dif_ini)
        form_right.addRow("Dificultad final", self.cb_dif_fin)
        form_right.addRow("Idioma texto", self.cb_idioma)

        self.btn_generate_book = QPushButton("Generate book")
        self.btn_generate_book.setEnabled(False)
        self.btn_generate_book.clicked.connect(self.on_generate_book)
        left.addWidget(self.btn_generate_book)
        self.btn_generate_book.hide()

        self.lbl_status = QLabel("")
        left.addWidget(self.lbl_status)

        self.table = QTableWidget(0, len(self.DISPLAY_COLS))
        self.table.setHorizontalHeaderLabels(self.DISPLAY_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        difficulty_col = self.DISPLAY_COLS.index("Nivel dificultad")
        self.table.horizontalHeader().setSectionResizeMode(
            difficulty_col, QHeaderView.Fixed
        )
        self.table.setColumnWidth(difficulty_col, 90)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        self.table.cellClicked.connect(self.on_table_cell_clicked)
        self.table.cellDoubleClicked.connect(self.on_table_double_clicked)
        left.addWidget(self.table, 1)

        # Create icons for clickable/disabled
        self._icon_ok_pix = QPixmap(20, 20)
        self._icon_ok_pix.fill(Qt.transparent)
        self._icon_ok_pix = self.draw_icon_image_ok()
        self._icon_gray_pix = QPixmap(20, 20)
        self._icon_gray_pix.fill(Qt.transparent)
        self._icon_gray_pix = self.draw_icon_image_disabled()
        self._icon_ok = QIcon(self._icon_ok_pix)
        self._icon_disabled = QIcon(self._icon_gray_pix)

        right_box = QGroupBox("Selected illustration")
        right_box.setMinimumWidth(420)
        right_box.setMaximumWidth(420)
        right.addWidget(right_box)
        rvl = QVBoxLayout(right_box)


        # --- PROMPT BLOCK ---
        prompt_box = QGroupBox("Prompt (final + negative)")
        prompt_layout = QVBoxLayout(prompt_box)

        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setReadOnly(True)
        prompt_layout.addWidget(self.txt_prompt)

        rvl.addWidget(prompt_box, 2)

        # --- EDITORIAL TEXT BLOCK ---
        editorial_box = QGroupBox("Inspirational text")
        editorial_layout = QVBoxLayout(editorial_box)

        self.txt_editorial = QPlainTextEdit()
        self.txt_editorial.setReadOnly(True)
        editorial_layout.addWidget(self.txt_editorial)

        rvl.addWidget(editorial_box, 1)

        # --- IMAGE PREVIEW BLOCK ---
        image_box = QGroupBox("Image preview")
        image_layout = QVBoxLayout(image_box)

        self.lbl_image_preview = QLabel("No image generated")
        self.lbl_image_preview.setAlignment(Qt.AlignCenter)
        self.lbl_image_preview.setMinimumHeight(260)
        self.lbl_image_preview.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                color: #555;
                border: 1px dashed #aaa;
            }
        """)
        self.lbl_image_preview.setScaledContents(True)

        image_layout.addWidget(self.lbl_image_preview)
        rvl.addWidget(image_box, 3)


        self.btn_generate_image = QPushButton("Generate image for selected row")
        self.btn_generate_image.setEnabled(False)
        self.btn_generate_image.clicked.connect(self.on_generate_image)
        rvl.addWidget(self.btn_generate_image)
        self.btn_generate_image.hide()

        self.lbl_image_status = QLabel("")
        rvl.addWidget(self.lbl_image_status)

        self.ed_book_id.textChanged.connect(self.validate_form)
        self.ed_tema.textChanged.connect(self.validate_form)
        self.ed_alcance.textChanged.connect(self.validate_form)
        self.validate_form()

    # ---- Menu action enablement helpers ----
    def validate_form(self):
        has_project = self.project_dir is not None
        ok = (
            self.ed_book_id.text().strip() != ""
            and self.ed_tema.text().strip() != ""
            and self.ed_alcance.text().strip() != ""
        )
        self.btn_generate_book.setEnabled(ok)
        # Sync menu action enable state:
        if hasattr(self, "action_generar_libro"):
            self.action_generar_libro.setEnabled(ok)

    def on_row_selected(self):
        sel = self.table.selectionModel().selectedRows()
        ok = bool(sel)
        self.btn_generate_image.setEnabled(ok)
        if hasattr(self, "action_generar_imagen"):
            self.action_generar_imagen.setEnabled(ok)
        # You may have other logic here regarding updating selection text, etc.
        if not sel:
            self.txt_prompt.clear()
            self.txt_editorial.clear()
            self.lbl_image_preview.setText("No image generated")
            self.lbl_image_preview.setPixmap(QPixmap())
            return

        row = self.rows[sel[0].row()]

        # Prompt completo
        self.txt_prompt.setPlainText(
            row.Prompt_final.strip() + "\n\n--- NEGATIVE PROMPT ---\n\n" + row.Prompt_negativo.strip()
        )

        # Texto editorial
        self.txt_editorial.setPlainText(row.Comentario_editorial.strip())

        # Imagen
        img_base = self.images_dir if self.images_dir else OUT_DIR
        img_path = img_base / f"{row.ID}.png"

        if img_path.exists():
            pix = QPixmap(str(img_path))
            self.lbl_image_preview.setPixmap(pix)
            self.lbl_image_preview.setStyleSheet("background-color: black;")
        else:
            self.lbl_image_preview.setPixmap(QPixmap())
            self.lbl_image_preview.setText("No image generated")
            self.lbl_image_preview.setStyleSheet("""
                QLabel {
                    background-color: #e0e0e0;
                    color: #555;
                    border: 1px dashed #aaa;
                }
            """)

    # ----------------- NEW: Open/Load Project Logic ---------------------
    def on_open_existing_book(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        dir_path = QFileDialog.getExistingDirectory(self, "Selecciona carpeta del libro", str(PROJECTS_DIR))
        if not dir_path:
            return

        project_dir = Path(dir_path)
        book_json = project_dir / "book.json"
        illustrations_json = project_dir / "illustrations.json"
        images_dir = project_dir / "images"

        # Validation
        if not book_json.exists():
            QMessageBox.critical(self, "Error", f"Falta book.json en:\n{project_dir}")
            return
        if not illustrations_json.exists():
            QMessageBox.critical(self, "Error", f"Falta illustrations.json en:\n{project_dir}")
            return
        if not images_dir.exists():
            try:
                images_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo crear images/:\n{e}")
                return

        self.load_project(project_dir)

    def load_project(self, project_dir):
        from PySide6.QtWidgets import QMessageBox

        import json

        book_json_p = project_dir / "book.json"
        illus_json_p = project_dir / "illustrations.json"
        images_dir = project_dir / "images"

        # Parse book.json
        try:
            with book_json_p.open("r", encoding="utf-8") as f:
                book_data = json.load(f)
                spec = book_data.get("spec", {})

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer book.json:\n{e}")
            return

        # Restore book form fields
        try:
            # Handle missing keys gracefully if needed
            self.ed_book_id.setText(spec.get("book_id", ""))
            self.cb_publico.setCurrentText(spec.get("publico", "Adultos"))
            self.ed_tema.setText(spec.get("tema", ""))
            self.ed_alcance.setText(spec.get("alcance_geografico", ""))
            self.ed_tipo.setText(spec.get("tipo_imagenes", ""))
            self.sp_num.setValue(spec.get("numero_imagenes", 0))
            self.ed_tamano.setText(spec.get("tamano_libro", "8.5x11"))
            self.ed_aspect.setText(spec.get("aspect_ratio", "4:5"))
            self.cb_dif_ini.setCurrentText(spec.get("dificultad_inicial", "Bajo"))
            self.cb_dif_fin.setCurrentText(spec.get("dificultad_final", "Alto"))
            self.cb_idioma.setCurrentText(spec.get("idioma_texto", "es"))


            # Text/image model
            text_model = spec.get("text_model", DEFAULT_TEXT_MODEL)
            idx = self.cb_text_model.findText(text_model)
            if idx != -1:
                self.cb_text_model.setCurrentIndex(idx)
            else:
                self.cb_text_model.setCurrentIndex(0)

            image_model = spec.get("image_model", DEFAULT_IMAGE_MODEL)
            idx = self.cb_image_model.findText(image_model)
            if idx != -1:
                self.cb_image_model.setCurrentIndex(idx)
            else:
                self.cb_image_model.setCurrentIndex(0)

            image_resolution = spec.get("image_resolution", "Testing (1024 x 1024)")
            idx = self.cb_image_resolution.findText(image_resolution)
            if idx != -1:
                self.cb_image_resolution.setCurrentIndex(idx)
            else:
                self.cb_image_resolution.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error restaurando campos de book.json:\n{e}")
            return

        # Set directories
        self.project_dir = project_dir
        self.images_dir = images_dir

        # Parse illustrations.json and repopulate table/rows
        try:
            with illus_json_p.open("r", encoding="utf-8") as f:
                illus_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer illustrations.json:\n{e}")
            return

        rows_raw = illus_data.get("illustrations", [])
        self.rows = []
        self.table.setRowCount(0)
        from pydantic import ValidationError
        self.rows = []
        for row_data in rows_raw:
            try:
                row = IllustrationRow(**row_data)

                # Backward compatibility for old projects
                if not getattr(row, "approval_status", None):
                    row.approval_status = "pending"

                if not hasattr(row, "rejection_reason"):
                    row.rejection_reason = ""

                self.rows.append(row)

            except ValidationError as e:
                QMessageBox.critical(
                    self,
                    "Invalid illustration data",
                    f"Error loading illustration:\n{e}"
                )
                return



            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)
            # Assuming columns match self.DISPLAY_COLS (first col is image status)
            # Set icon for image
            image_id = row.ID
            img_path = self.images_dir / f"{image_id}.png"
            icon = self._icon_ok if img_path.exists() else self._icon_disabled
            img_item = QTableWidgetItem()
            img_item.setIcon(icon)
            self.table.setItem(row_idx, 0, img_item)
            # Fill rest of columns from row dict
            for c_idx, (_, attr) in enumerate(self.TABLE_COLUMN_MAP, start=1):
                value = getattr(row, attr, "")
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(row_idx, c_idx, item)
            self._apply_row_color(row_idx, row.approval_status)


        # Update UI: project label, clear selection, reset editors
        slug = project_dir.name
        self.lbl_project.setText(f"Current book: {slug}\n{project_dir}")

        self.table.clearSelection()
        self.txt_prompt.clear()
        self.txt_editorial.clear()
        # Status labels
        self.lbl_status.setText("")
        self.lbl_image_status.setText("")

        self.validate_form()
        self.on_row_selected()
        self.validate_form()
        self.action_generar_libro.setEnabled(True)

    def on_save_project(self):
        # 1) Validar formulario
        try:
            spec = self.get_spec()
        except ValidationError as e:
            QMessageBox.critical(self, "Invalid inputs", str(e))
            return

        # 2) Si NO existe proyecto → pedir carpeta
        if self.project_dir is None:
            base_dir = QFileDialog.getExistingDirectory(
                self, "Select folder to save the project"
            )
            if not base_dir:
                return

            slug = _safe_slug(spec.book_id)
            project_dir = Path(base_dir) / slug
            images_dir = project_dir / "images"

            if project_dir.exists():
                QMessageBox.critical(
                    self,
                    "Folder exists",
                    f"Project folder already exists:\n{project_dir}",
                )
                return

            project_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            self.project_dir = project_dir
            self.images_dir = images_dir

        # 3) Guardar book.json
        (self.project_dir / "book.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.2",
                    "book_id": spec.book_id,
                    "spec": spec.model_dump(),
                    "image_settings": {
                        "image_model": self.get_selected_image_model(),
                        "image_resolution": self.get_selected_image_resolution(),
                    },
                    "text_settings": {
                        "text_model": self.get_selected_text_model(),
                    },
                    "last_modified_at": _utc_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # 4) Guardar illustrations.json
        (self.project_dir / "illustrations.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.2",
                    "illustrations": [r.model_dump() for r in self.rows],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.lbl_project.setText(
            f"Current book: {self.project_dir.name}\n{self.project_dir}"
        )

        QMessageBox.information(self, "Saved", "Project saved successfully.")

    def reset_project_state(self):
        self.project_dir = None
        self.images_dir = None
        self.rows = []

        self.lbl_project.setText("No book loaded")

        self.ed_book_id.clear()
        self.ed_tema.clear()
        self.ed_alcance.clear()
        self.ed_tipo.setText("Monumentos y paisajes")
        self.sp_num.setValue(50)

        self.table.setRowCount(0)
        self.table.clearSelection()

        self.txt_prompt.clear()
        self.txt_editorial.clear()

        self.lbl_status.setText("")
        self.lbl_image_status.setText("")

        self.validate_form()


    def draw_icon_image_ok(self):
        """Draw a green eye or document icon for image exists."""
        pix = QPixmap(20, 20)
        pix.fill(Qt.transparent)
        from PySide6.QtGui import QPainter, QPen, QBrush

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        # Draw green circle
        painter.setPen(QPen(QColor(60, 180, 90), 2))
        painter.setBrush(QBrush(Qt.white))
        painter.drawEllipse(2, 2, 16, 16)
        # Draw black center (simulate eye/pupil/file)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(7, 7, 6, 6)
        painter.end()
        return pix

    def draw_icon_image_disabled(self):
        """Draw a gray disabled eye or document icon."""
        pix = QPixmap(20, 20)
        pix.fill(Qt.transparent)
        from PySide6.QtGui import QPainter, QPen, QBrush

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        # Draw gray outer circle
        painter.setPen(QPen(QColor(120, 120, 120), 2))
        painter.setBrush(QBrush(QColor(220, 220, 220)))
        painter.drawEllipse(2, 2, 16, 16)
        # Draw gray center
        painter.setPen(QPen(QColor(140, 140, 140), 2))
        painter.drawEllipse(7, 7, 6, 6)
        painter.end()
        return pix

    def _set_req_label(self, label: QLabel, name: str, status: str):
        if status == "ok":
            label.setText(f"🟢 {name}: loaded")
        elif status == "empty":
            label.setText(f"🟡 {name}: file is empty")
        else:
            label.setText(f"🔴 {name}: file not found")




    def get_spec(self) -> BookSpec:
        return BookSpec(
            book_id=self.ed_book_id.text().strip(),
            publico=self.cb_publico.currentText(),
            tema=self.ed_tema.text().strip(),
            alcance_geografico=self.ed_alcance.text().strip(),
            tipo_imagenes=self.ed_tipo.text().strip(),
            numero_imagenes=int(self.sp_num.value()),
            tamano_libro=self.ed_tamano.text().strip(),
            aspect_ratio=self.ed_aspect.text().strip(),
            dificultad_inicial=self.cb_dif_ini.currentText(),
            dificultad_final=self.cb_dif_fin.currentText(),
            idioma_texto=self.cb_idioma.currentText(),
        )

    def get_selected_text_model(self) -> str:
        return self.cb_text_model.currentText().strip() or DEFAULT_TEXT_MODEL

    def get_selected_image_model(self) -> str:
        return self.cb_image_model.currentText().strip() or DEFAULT_IMAGE_MODEL

    def on_generate_book(self):
        try:
            spec = self.get_spec()
             # Advertir si ya existe contenido generado
            has_rows = bool(self.rows)
            has_images = self.images_dir and any(self.images_dir.glob("*.png"))

            if has_rows or has_images:
                reply = QMessageBox.warning(
                    self,
                    "Book already generated",
                    "This book already has generated content (table or images).\n\n"
                    "If you continue, all existing illustrations and images will be deleted.\n\n"
                    "Do you want to continue?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                )
                if reply != QMessageBox.Yes:
                    return

                # Reset contenido existente
                self.rows = []
                self.table.setRowCount(0)

                if self.images_dir and self.images_dir.exists():
                    for p in self.images_dir.glob("*.png"):
                        p.unlink(missing_ok=True)

        except ValidationError as e:
            QMessageBox.critical(self, "Invalid inputs", str(e))
            return
        # 1) Comprobar que hay un libro cargado
        if self.project_dir is None:
            QMessageBox .critical(
                self,
                "No book loaded",
                "Create or open a book first.",
            )
            return

        # 2) Comprobar que los requerimientos están cargados
        if (
            self.req_imagenes["status"] != "ok"
            or self.req_texto["status"] != "ok"
        ):
            QMessageBox.critical(
                self,
                "Editorial requirements missing",
                "Image and text requirement files must exist and not be empty "
                "before generating a book.",
            )
            return

        self.btn_generate_book.setEnabled(False)
        text_model = self.get_selected_text_model()
        self.lbl_status.setText("Generating book table via OpenAI...")
        self.overlay.start_indeterminate("🤖 AI is generating the book…")
        worker = GenerateBookWorker(
            self.client,
            spec,
            text_model,
            req_img=self.req_imagenes["content"],
            req_txt=self.req_texto["content"],
        )
        worker.signals.ok.connect(self.on_book_generated)
        worker.signals.err.connect(self.on_worker_error)
        worker.signals.progress.connect(self.overlay.update)
        self.pool.start(worker)
    def _apply_row_color(self, row_idx: int, status: str):
        if status == "approved":
            bg_color = QColor("#2f6f4e")   # verde oscuro elegante
            fg_color = QColor("#ffffff")   # texto blanco
        elif status == "rejected":
            bg_color = QColor("#7a2e2e")   # rojo oscuro elegante
            fg_color = QColor("#ffffff")   # texto blanco
        else:
            bg_color = None
            fg_color = None

        for col in range(self.table.columnCount()):
            item = self.table.item(row_idx, col)
            if not item:
                continue

            if bg_color:
                item.setBackground(QBrush(bg_color))
                item.setForeground(QBrush(fg_color))
            else:
                item.setBackground(QBrush())
                item.setForeground(QBrush())  # vuelve al color por defecto





    def on_book_generated(self, rows: List[IllustrationRow]):
        self.overlay.stop()
        self.btn_generate_book.setEnabled(True)
        self.rows = rows
        self.table.setRowCount(len(rows))
        # Image icons go in column 0
        for r_idx, row in enumerate(rows):
            d = row.model_dump()
            img_base = self.images_dir if self.images_dir is not None else OUT_DIR
            img_path = img_base / f"{row.ID}.png"

            image_exists = img_path.exists()
            icon_item = QTableWidgetItem()
            if image_exists:
                icon_item.setIcon(self._icon_ok)
                icon_item.setToolTip("Image generated")

                # Set as clickable (normal enabled flags)
                icon_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            else:
                icon_item.setIcon(self._icon_disabled)
                icon_item.setToolTip("Image not found")
                # Set as enabled (but we handle click as do-nothing), or can set ~enabled
                icon_item.setFlags(Qt.NoItemFlags)

            # Slightly center the icon
            icon_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r_idx, 0, icon_item)

            for c_idx, (_, attr) in enumerate(self.TABLE_COLUMN_MAP, start=1):
                value = getattr(row, attr, "")
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(r_idx, c_idx, item)
            self._apply_row_color(r_idx, row.approval_status)


        self.lbl_status.setText(f"Book generated: {len(rows)} rows")
        if self.project_dir is not None:
            spec = self.get_spec()
            (self.project_dir / "book.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.2",
                        "book_id": spec.book_id,
                        "spec": spec.model_dump(),
                        "image_settings": {
                            "image_model": self.get_selected_image_model(),
                            "image_resolution": self.get_selected_image_resolution(),
                        },
                        "text_settings": {"text_model": self.get_selected_text_model()},
                        "last_modified_at": _utc_now_iso(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            (self.project_dir / "illustrations.json").write_text(
                json.dumps(
                    {"schema_version": "1.2", "illustrations": [r.model_dump() for r in self.rows]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )



    def on_image_generated(self, out_path: str):
        self.overlay.stop()
        self.btn_generate_image.setEnabled(True)
        self.lbl_image_status.setText(f"Saved: {out_path}")

        # Refrescar icono de la fila
        for r_idx, row in enumerate(self.rows):
            if out_path.endswith(f"{row.ID}.png"):
                item = QTableWidgetItem()
                item.setIcon(self._icon_ok)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r_idx, 0, item)
                break
        self.on_row_selected()


    def on_generate_image(self):
        sel = self.table.selectionModel().selectedRows()
        self.btn_generate_image.setEnabled(False)
        if not sel:
            return
        row = self.rows[sel[0].row()]
        if row.approval_status == "approved":
            reply = QMessageBox.warning(
                self,
                "Image already approved",
                "This image is already approved.\n\n"
                "Regenerating it will discard the approved image.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        image_model = self.get_selected_image_model()
        self.overlay.start_indeterminate("🎨 AI is generating the image…")
        image_resolution = self.get_selected_image_resolution()

       
        worker = GenerateImageWorker(
            self.client,
            row,
            image_model,
            image_resolution,
            self.images_dir,
        )
        worker.signals.progress.connect(self.overlay.update)
        worker.signals.ok.connect(self.on_image_generated)
        worker.signals.err.connect(self.on_worker_error)
        self.pool.start(worker)

    def on_worker_error(self, msg: str):
        self.overlay.stop()
        self.btn_generate_book.setEnabled(True)
        self.btn_generate_image.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)
    def on_table_cell_clicked(self, row_idx, col_idx):
        return

    def on_table_double_clicked(self, row_idx, col_idx):
        if not (0 <= row_idx < len(self.rows)):
            return

        row = self.rows[row_idx]
        img_base = self.images_dir if self.images_dir else OUT_DIR
        img_path = img_base / f"{row.ID}.png"

        if not img_path.exists():
            reply = QMessageBox.question(
                self,
                "Generate image",
                "This image has not been generated yet.\nDo you want to generate it now?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.table.selectRow(row_idx)
                self.on_generate_image()
            return

        row = self.rows[row_idx]
        img_path = self.images_dir / f"{row.ID}.png"

        dlg = ImageReviewDialog(self, row, img_path)
        if dlg.exec():
            self._apply_row_color(row_idx, row.approval_status)
            self.on_save_project()




def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
