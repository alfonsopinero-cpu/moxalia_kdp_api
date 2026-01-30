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

)
from PySide6.QtGui import QIcon, QPixmap, QColor, QBrush, QCursor, QDesktopServices
from PySide6.QtCore import QUrl

# =========================
# CONFIG
# =========================
load_dotenv()

IMAGE_RESOLUTIONS = {
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
    ):
        super().__init__()
        self.client = client
        self.row = row
        self.image_model = image_model
        self.image_resolution = image_resolution
        self.signals = WorkerSignals()

    def run(self):
        try:
            out_path = OUT_DIR / f"{self.row.ID}.png"
            generate_image_png(
                self.client,
                prompt=self.row.Prompt_final,
                negative=self.row.Prompt_negativo,
                out_path=out_path,
                image_model=self.image_model,
                image_resolution=self.image_resolution,
                progress_cb=lambda p, m: self.signals.progress.emit(p, m)
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
    DISPLAY_COLS = ["Image"] + [
        "ID",
        "Categoria",
        "Publico",
        "Pais_Region",
        "Monumento",
        "Main_foco",
        "Difficulty_block",
        "Difficulty_D",
        "Nombre_fichero_editorial",
        "Nombre_fichero_descargado",
        "Comentario_editorial",
        "Prompt_final",
        "Prompt_negativo",
    ]
    COLS = DISPLAY_COLS[1:]  # For mapping/row dict only

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KDP Coloring Book Builder (MVP)")
        self.resize(1400, 800)

        self.client = OpenAI()
        self.pool = QThreadPool.globalInstance()
        self.overlay = BlockingOverlay(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.hide()
        self.req_imagenes = load_requirement_file(REQ_IMAGENES_PATH)
        self.req_texto = load_requirement_file(REQ_TEXTO_PATH)


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

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QVBoxLayout()
        main.addLayout(left, 3)

        right = QVBoxLayout()
        main.addLayout(right, 2)

        form_box = QGroupBox("Book inputs")
        req_box = QGroupBox("Editorial requirements")
        left.addWidget(req_box)
        req_layout = QVBoxLayout(req_box)

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
        form = QFormLayout(form_box)

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
        form.addRow("Text model", self.cb_text_model)

        self.cb_image_resolution = QComboBox()
        self.cb_image_resolution.addItems(IMAGE_RESOLUTIONS.keys())
        self.cb_image_resolution.setCurrentText("Working (1024 x 1536)")
        form.addRow("Image resolution", self.cb_image_resolution)

        self.cb_image_model = QComboBox()
        for m in self.available_image_models:
            self.cb_image_model.addItem(m)
        idx_image = self.cb_image_model.findText(DEFAULT_IMAGE_MODEL)
        if idx_image != -1:
            self.cb_image_model.setCurrentIndex(idx_image)
        form.addRow("Image model", self.cb_image_model)
        form.addRow("Book ID", self.ed_book_id)
        form.addRow("Publico", self.cb_publico)
        form.addRow("Tema", self.ed_tema)
        form.addRow("Alcance geográfico", self.ed_alcance)
        form.addRow("Tipo de imágenes", self.ed_tipo)
        form.addRow("Número de imágenes", self.sp_num)
        form.addRow("Tamaño libro", self.ed_tamano)
        form.addRow("Aspect ratio", self.ed_aspect)
        form.addRow("Dificultad inicial", self.cb_dif_ini)
        form.addRow("Dificultad final", self.cb_dif_fin)
        form.addRow("Idioma texto", self.cb_idioma)

        self.btn_generate_book = QPushButton("Generate book")
        self.btn_generate_book.setEnabled(False)
        self.btn_generate_book.clicked.connect(self.on_generate_book)
        left.addWidget(self.btn_generate_book)

        self.lbl_status = QLabel("")
        left.addWidget(self.lbl_status)

        self.table = QTableWidget(0, len(self.DISPLAY_COLS))
        self.table.setHorizontalHeaderLabels(self.DISPLAY_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        self.table.cellClicked.connect(self.on_table_cell_clicked)
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
        right.addWidget(right_box)
        rvl = QVBoxLayout(right_box)

        self.sel_title = QLabel("No row selected")
        self.sel_title.setWordWrap(True)
        rvl.addWidget(self.sel_title)

        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setReadOnly(True)
        rvl.addWidget(self.txt_prompt, 2)

        self.txt_editorial = QPlainTextEdit()
        self.txt_editorial.setReadOnly(True)
        rvl.addWidget(self.txt_editorial, 1)


        self.btn_generate_image = QPushButton("Generate image for selected row")
        self.btn_generate_image.setEnabled(False)
        self.btn_generate_image.clicked.connect(self.on_generate_image)
        rvl.addWidget(self.btn_generate_image)

        self.lbl_image_status = QLabel("")
        rvl.addWidget(self.lbl_image_status)

        self.ed_book_id.textChanged.connect(self.validate_form)
        self.ed_tema.textChanged.connect(self.validate_form)
        self.ed_alcance.textChanged.connect(self.validate_form)
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

    def validate_form(self):
        ok = (
            self.ed_book_id.text().strip() != ""
            and self.ed_tema.text().strip() != ""
            and self.ed_alcance.text().strip() != ""
            and self.sp_num.value() > 0
        )
        self.btn_generate_book.setEnabled(ok)

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
        except ValidationError as e:
            QMessageBox.critical(self, "Invalid inputs", str(e))
            return
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

    def on_book_generated(self, rows: List[IllustrationRow]):
        self.overlay.stop()
        self.btn_generate_book.setEnabled(True)
        self.rows = rows
        self.table.setRowCount(len(rows))
        # Image icons go in column 0
        for r_idx, row in enumerate(rows):
            d = row.model_dump()
            img_path = OUT_DIR / f"{row.ID}.png"
            image_exists = img_path.exists()
            icon_item = QTableWidgetItem()
            if image_exists:
                icon_item.setIcon(self._icon_ok)
                icon_item.setToolTip(f"Click to open {img_path.name}")
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

            for c_idx, key in enumerate(self.COLS):
                item = QTableWidgetItem(str(d.get(key, "")))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(r_idx, c_idx+1, item)

        self.lbl_status.setText(f"Book generated: {len(rows)} rows")
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
    def on_row_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        # Our data rows start at column 1 (after "Image" column)
        row_idx = sel[0].row()
        row = self.rows[row_idx]
        self.sel_title.setText(f"Row {row_idx+1}: {row.ID} - {row.Categoria} - {row.Pais_Region}")
        combined_prompt = (
            "=== IMAGE PROMPT ===\n\n"
            f"{row.Prompt_final}\n\n"
            "--- NEGATIVE CONSTRAINTS ---\n"
            f"{row.Prompt_negativo}"
        )
        self.txt_prompt.setPlainText(combined_prompt)

        self.txt_editorial.setPlainText(
            "=== INSPIRATIONAL TEXT ===\n\n"
            f"{row.Comentario_editorial}"
        )

        self.btn_generate_image.setEnabled(True)

    def on_generate_image(self):
        sel = self.table.selectionModel().selectedRows()
        self.btn_generate_image.setEnabled(False)
        if not sel:
            return
        row = self.rows[sel[0].row()]

        image_model = self.get_selected_image_model()
        self.overlay.start_indeterminate("🎨 AI is generating the image…")
        image_resolution = self.get_selected_image_resolution()

        worker = GenerateImageWorker(
            self.client,
            row,
            image_model,
            image_resolution,
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
        # Only respond to the icon column (col_idx == 0)
        if col_idx != 0:
            return
        # Defensive: row index should be valid and row exists
        if not (0 <= row_idx < len(self.rows)):
            return
        illustration_row = self.rows[row_idx]
        img_path = OUT_DIR / f"{illustration_row.ID}.png"
        if img_path.exists():
            # Open with default app
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(img_path)))

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
