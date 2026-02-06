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

from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, QSize, QEvent, QTimer
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
    QGraphicsView,
    QGraphicsScene,
    QGraphicsTextItem,

)
from PySide6.QtGui import QIcon, QPixmap, QColor, QBrush, QCursor, QDesktopServices, QAction, QPainter, QWheelEvent, QKeyEvent, QKeySequence, QShortcut
from PySide6.QtCore import QUrl
from datetime import datetime

# =========================
# CONFIG
# =========================
load_dotenv()


PROMPTS_DIR = Path.cwd() / "prompts"

REQ_IMAGENES_PATH = PROMPTS_DIR / "requerimientos_imagenes.txt"
REQ_TEXTO_PATH = PROMPTS_DIR / "requerimientos_texto.txt"
SYSTEM_PATH = PROMPTS_DIR / "system.txt"
USER_PATH = PROMPTS_DIR / "user.txt"

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")


OUT_DIR = Path.cwd() / "illustraciones"
OUT_DIR.mkdir(parents=True, exist_ok=True)


LOGS_DIR = Path.cwd() / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_LOG_CSV = LOGS_DIR / "openai_calls_log.csv"

OPENAI_LOG_CSV_V2 = LOGS_DIR / "openai_calls_log_v2.csv"
PRICING_VERSION = "2026-02"

STATE_PATH = Path.cwd() / "book_state.json"

PROJECTS_DIR = Path.cwd() / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_MODEL_CAPABILITIES = {
    "gpt-image-1": {
        "qualities": ["low", "medium", "high"],
        "sizes": ["1024x1024", "1024x1536", "1536x1024"],
        "default_quality": "low",
        "default_size": "1024x1536",
    },
    "gpt-image-1-mini": {
        "qualities": ["low", "medium", "high"],
        "sizes": ["1024x1024", "1024x1536", "1536x1024"],
        "default_quality": "low",
        "default_size": "1024x1536",
    },
    "gpt-image-1.5": {
        "qualities": ["low", "medium", "high"],
        "sizes": ["1024x1024", "1024x1536", "1536x1024"],
        "default_quality": "low",
        "default_size": "1024x1536",
    },
    "dall-e-3": {
        "qualities": ["standard", "hd"],
        "sizes": ["1024x1024", "1024x1792", "1792x1024"],
        "default_quality": "standard",
        "default_size": "1024x1792",
    },
}

# =========================
# DATA MODELS
# =========================
from typing import Literal

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

    image_quality: Literal["low", "medium", "high"] = "low"



class IllustrationRow(BaseModel):
    ID: str
    Categoria: str
    Publico: str
    Pais_Region: str
    Monumento: str
    Main_foco: str
    Bloque: str
    Difficulty_block: str
    Difficulty_D: int
    Line_thickness_L: float
    White_space_W: float
    Complexity_C: float

    Prompt_final: str
    Prompt_negativo: str
    Comentario_editorial: str

    Nombre_fichero_editorial: str
    Nombre_fichero_descargado: str
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




# =========================
# OPENAI HELPERS
# =========================
import json

PRICING_PATH = Path("pricing.json")


# Load pricing once at startup
if not PRICING_PATH.exists():
    raise RuntimeError("pricing.json not found")

with PRICING_PATH.open("r", encoding="utf-8") as f:
    PRICING_DATA = json.load(f)



def estimate_cost(
    *,
    pricing: dict,
    model: str,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    image_size: str | None = None,
    image_quality: str | None = None,
) -> float:


    models = pricing.get("models", {})
    if model not in models:
        return 0.0

    model_cfg = models[model]

    # TEXT MODELS
    if model_cfg["type"] == "text":
        if tokens_input is None or tokens_output is None:
            return 0.0

        return (
            tokens_input * model_cfg["input_token"]
            + tokens_output * model_cfg["output_token"]
        )

    # IMAGE MODELS
    if model_cfg["type"] == "image":
        if image_size is None or image_quality is None:
            return 0.0

        return model_cfg["prices"][image_quality].get(image_size, 0.0)



    return 0.0


import uuid
import csv

def log_openai_call_v2(
    *,
    request_id: str,
    call_type: str,
    model: str,
    start_ts: str,
    end_ts: str,
    status: str,
    phase: str,
    user_action: str,
    project_id: str | None,
    book_id: str | None,
    row_id: str | None,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    image_size: str | None = None,
    image_quality: str | None = None,
    retry_count: int = 0,
    approval_outcome: str = "na",
    cost_estimated: float | None = None,
    error: str = "",
):
    file_exists = OPENAI_LOG_CSV_V2.exists()

    latency_ms = (
        int(
            (datetime.fromisoformat(end_ts) - datetime.fromisoformat(start_ts))
            .total_seconds() * 1000
        )
        if start_ts and end_ts else None
    )
    
   

    cost_estimated = estimate_cost(
        pricing=PRICING_DATA,
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        image_size=image_size,
        image_quality=image_quality,
    )




    with OPENAI_LOG_CSV_V2.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp_start",
                "timestamp_end",
                "latency_ms",
                "request_id",
                "project_id",
                "book_id",
                "row_id",
                "phase",
                "user_action",
                "type",
                "model",
                "status",
                "tokens_input",
                "tokens_output",
                "image_size",
                "image_quality",
                "retry_count",
                "approval_outcome",
                "pricing_version",
                "cost_estimated",
                "error",

            ])


        writer.writerow([
            start_ts,
            end_ts,
            latency_ms,
            request_id,
            project_id,
            book_id,
            row_id,
            phase,
            user_action,
            call_type,
            model,
            status,
            tokens_input,
            tokens_output,
            image_size,
            image_quality,
            retry_count,
            approval_outcome,
            PRICING_VERSION,
            cost_estimated,
            error,
        ])


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
    
def _validate_batch_rows(data: Any, expected_n: int) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError("Model did not return a JSON array")
    if len(data) != expected_n:
        raise ValueError(f"Batch row count mismatch: expected {expected_n}, got {len(data)}")

    required = {
        "Categoria",
        "Publico",
        "Pais_Region",
        "Monumento",
        "Main_foco",
        "Prompt_final",
        "Prompt_negativo",
        "Comentario_editorial",
    }
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"Batch item {i} is not an object")
        missing = required - set(obj.keys())
        if missing:
            raise ValueError(f"Batch item {i} missing keys: {sorted(missing)}")
    return data


def _reorder_no_consecutive_countries(rows: list["IllustrationRow"]) -> list["IllustrationRow"]:
    rows = list(rows)
    for i in range(1, len(rows)):
        if rows[i].Pais_Region == rows[i - 1].Pais_Region:
            swap_j = None
            for j in range(i + 1, len(rows)):
                if rows[j].Pais_Region != rows[i - 1].Pais_Region:
                    swap_j = j
                    break
            if swap_j is not None:
                rows[i], rows[swap_j] = rows[swap_j], rows[i]
    return rows
def _norm_scope(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

MULTI_COUNTRY_SCOPES = {
    # ES
    "europa", "asia", "america", "américa", "latam", "latinoamerica", "latinoamérica",
    "norte america", "norteamérica", "sudamerica", "sudamérica", "centroamerica", "centroamérica",
    "africa", "áfrica", "oceania", "oceanía",
    "global", "mundial", "mundo", "world", "international", "internacional",
    "europe", "asia pacific", "apac", "emea", "americas",
    # EN
    "europe", "asia", "africa", "oceania", "americas", "north america", "south america",
    "central america", "latin america", "global", "worldwide"
}

def _is_single_country_scope(alcance_geografico: str) -> bool:
    s = _norm_scope(alcance_geografico)
    if not s:
        return False
    # Si es un scope “continente/global”, NO es single-country
    return s not in MULTI_COUNTRY_SCOPES

    
def monument_importance_score(row: IllustrationRow) -> int:
    name = row.Monumento.lower()

    PRIORITY = [
        "eiffel", "colosseum", "great wall", "taj mahal",
        "petra", "machu picchu", "angkor", "sagrada familia",
        "acropolis", "pyramids",
    ]

    for i, key in enumerate(PRIORITY):
        if key in name:
            return 100 - i

    return 10

def generate_illustrations(
    client: OpenAI,
    spec: BookSpec,
    text_model: str,
    *,
    req_img: str,
    req_txt: str,
    progress_cb=None,
) -> List[IllustrationRow]:

    total = spec.numero_imagenes
    batch_size = 5
    num_batches = (total + batch_size - 1) // batch_size

    rows_acc: List[IllustrationRow] = []
    system = SYSTEM_PATH.read_text(encoding="utf-8")
    user_template = USER_PATH.read_text(encoding="utf-8")

    used_monuments: set[str] = set()

    single_country_mode = _is_single_country_scope(spec.alcance_geografico)
    # En multi-country: geo_key = país
    # En single-country: geo_key = región interna (departamento/estado/comunidad/etc)
    geo_counter: dict[str, int] = {}
    MAX_PER_GEO = 4

    for b in range(num_batches):
        start_idx = b * batch_size
        n = min(batch_size, total - start_idx)
        # 🔵 PROGRESS: inicio de batch
        if progress_cb:
            progress_cb(
                int((b / num_batches) * 100),
                f"Generating batch {b+1}/{num_batches}"
            )


        user = user_template.format(
            book_spec=json.dumps(spec.model_dump(), ensure_ascii=False, indent=2),
            req_img=req_img,
            req_txt=req_txt,
            batch_size=n,
            blocked_countries=json.dumps(
                [k for k, cnt in geo_counter.items() if cnt >= MAX_PER_GEO],
                ensure_ascii=False,
            ),
            used_monuments=json.dumps(list(used_monuments), ensure_ascii=False),
            country_counts=json.dumps(geo_counter, ensure_ascii=False),

        )
        if single_country_mode:
            country_name = spec.alcance_geografico.strip()
            user += (
                "\n\nCRITICAL GEOGRAPHY RULE (NON-NEGOTIABLE):\n"
                f"- This is a SINGLE-COUNTRY book: '{country_name}'.\n"
                "- The field 'Pais_Region' MUST contain an internal region within that country "
                "(state/department/province/community/area), NOT the country name.\n"
                "- Do NOT repeat the same region more than allowed.\n"
            )


        request_id = str(uuid.uuid4())
        start_ts = _utc_now_iso()

        try:
            r = client.chat.completions.create(
                model=text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )

            end_ts = _utc_now_iso()
            usage = r.usage

            log_openai_call_v2(
                request_id=request_id,
                call_type="generate_book_batch",
                model=text_model,
                start_ts=start_ts,
                end_ts=end_ts,
                status="success",
                phase="book_planning",
                user_action="batch",
                project_id=spec.book_id,
                book_id=spec.book_id,
                row_id=None,
                tokens_input=usage.prompt_tokens,
                tokens_output=usage.completion_tokens,
            )

        except Exception as e:
            end_ts = _utc_now_iso()

            log_openai_call_v2(
                request_id=request_id,
                call_type="generate_book_batch",
                model=text_model,
                start_ts=start_ts,
                end_ts=end_ts,
                status="error",
                phase="book_planning",
                user_action="batch",
                project_id=spec.book_id,
                book_id=spec.book_id,
                row_id=None,
                error=str(e),
            )
            raise

        data = _extract_json(r.choices[0].message.content)
        data = _validate_batch_rows(data, n)

        for obj in data:
            monument = obj["Monumento"].strip()
            geo = obj["Pais_Region"].strip()

            # Si es single-country y el modelo devuelve el país en vez de una región: descártalo
            if single_country_mode:
                if _norm_scope(geo) == _norm_scope(spec.alcance_geografico):
                    continue

            if monument in used_monuments or geo_counter.get(geo, 0) >= MAX_PER_GEO:
                continue  # saltamos pero NO contamos como fila válida

            row = IllustrationRow(
                ID=f"{len(rows_acc)+1:03d}",
                Categoria=obj["Categoria"],
                Publico=obj["Publico"],
                Pais_Region=geo,
                Monumento=monument,
                Main_foco=obj["Main_foco"],
                Bloque="",
                Difficulty_block="",
                Difficulty_D=0,
                Line_thickness_L=0.0,
                White_space_W=0.0,
                Complexity_C=0.0,
                Prompt_final=obj["Prompt_final"],
                Prompt_negativo=obj["Prompt_negativo"],
                Comentario_editorial=obj["Comentario_editorial"],
                Nombre_fichero_editorial="",
                Nombre_fichero_descargado="",
            )

            rows_acc.append(row)
            used_monuments.add(monument)
            geo_counter[geo] = geo_counter.get(geo, 0) + 1

    if len(rows_acc) < total:
        raise RuntimeError(
            f"Generated only {len(rows_acc)} rows out of {total}. "
            "Model output filtered too aggressively."
        )
    # =========================
    # ASSIGN DIFFICULTY SEQUENCE
    # =========================
    difficulty_seq = build_difficulty_sequence(
        total=len(rows_acc),
        start=spec.dificultad_inicial,
        end=spec.dificultad_final,
    )
    for row, diff in zip(rows_acc, difficulty_seq):
        row.Difficulty_D = diff["D"]
        row.Difficulty_block = diff["block"]
        row.Bloque = diff["block"]
        row.Line_thickness_L = diff["L"]
        row.White_space_W = diff["W"]
        row.Complexity_C = diff["C"]

    return rows_acc

def generate_image_png(
    *,
    client: OpenAI,
    prompt: str,
    negative: str,
    out_path: Path,
    image_model: str,
    image_quality: str,
    image_resolution: str,
    project_id: str,
    progress_cb=None,
) -> None:


    final_prompt = f"{prompt}\n\nNEGATIVE PROMPT:\n{negative}"

    request_id = str(uuid.uuid4())
    start_ts = _utc_now_iso()

    try:

        img = client.images.generate(
            model=image_model,
            prompt=final_prompt,
            size=image_resolution,
            quality=image_quality,
        )


        end_ts = _utc_now_iso()

        log_openai_call_v2(
            request_id=request_id,
            call_type="generate_image",
            model=image_model,
            start_ts=start_ts,
            end_ts=end_ts,
            status="success",
            phase="image_generation",
            user_action="single",
            project_id=project_id,
            book_id=project_id,
            row_id=out_path.stem,
            image_size=image_resolution,
            image_quality=image_quality,
        )

        out_path.write_bytes(base64.b64decode(img.data[0].b64_json))

    except Exception as e:
        end_ts = _utc_now_iso()

        log_openai_call_v2(
            request_id=request_id,
            call_type="generate_image",
            model=image_model,
            start_ts=start_ts,
            end_ts=end_ts,
            status="error",
            phase="image_generation",
            user_action="single",
            project_id=project_id,
            book_id=project_id,
            row_id=out_path.stem,
            image_size=image_resolution,
            error=str(e),
        )
        raise


def qc_detect_black_bg_or_frame(png_path: Path) -> list[str]:
    from PIL import Image
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    px = img.load()

    issues = []

    # --- check corners for non-white background (simple but effective) ---
    corners = [
        (5, 5), (w - 6, 5), (5, h - 6), (w - 6, h - 6)
    ]
    for (x, y) in corners:
        r, g, b = px[x, y]
        if r < 240 or g < 240 or b < 240:
            issues.append("Background is not pure white (corner pixels are dark).")
            break

    # --- check for rectangular border/frame: dark pixels along all 4 edges ---
    def edge_dark_ratio(samples):
        dark = 0
        for (x, y) in samples:
            r, g, b = px[x, y]
            if r < 80 and g < 80 and b < 80:
                dark += 1
        return dark / max(1, len(samples))

    step = max(1, min(w, h) // 200)
    top = [(x, 2) for x in range(0, w, step)]
    bottom = [(x, h - 3) for x in range(0, w, step)]
    left = [(2, y) for y in range(0, h, step)]
    right = [(w - 3, y) for y in range(0, h, step)]

    if (
        edge_dark_ratio(top) > 0.35 and
        edge_dark_ratio(bottom) > 0.35 and
        edge_dark_ratio(left) > 0.35 and
        edge_dark_ratio(right) > 0.35
    ):
        issues.append("Detected a rectangular border/frame (dark pixels on all edges).")

    return issues

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
            self.signals.progress.emit(0, "Starting batches (0%)")
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


def _next_error_filename(base_dir: Path, image_id: str) -> Path:
    """
    Find next error file name:
    001-error_1.png
    001-error_2.png
    """
    i = 1
    while True:
        p = base_dir / f"{image_id}-error_{i}.png"
        if not p.exists():
            return p
        i += 1
def estimate_book_cost(
    *,
    pricing: dict,
    text_model: str,
    image_model: str,
    image_quality: str,
    image_resolution: str,
    num_images: int,
    batch_size: int = 5,
    est_tokens_in: int = 3000,
    est_tokens_out: int = 1000,
) -> dict:
    text_calls = (num_images + batch_size - 1) // batch_size

    text_cost = text_calls * estimate_cost(
        pricing=pricing,
        model=text_model,
        tokens_input=est_tokens_in,
        tokens_output=est_tokens_out,
    )

    image_cost = num_images * estimate_cost(
        pricing=pricing,
        model=image_model,
        image_quality=image_quality,
        image_size=image_resolution,
    )

    return {
        "text_calls": text_calls,
        "image_calls": num_images,
        "text_cost": round(text_cost, 4),
        "image_cost": round(image_cost, 4),
        "total_cost": round(text_cost + image_cost, 4),
    }


class GenerateImageWorker(QRunnable):
    def __init__(
        self,
        client: OpenAI,
        row: IllustrationRow,
        image_model: str,
        image_quality: str,
        image_resolution: str,
        images_dir: Path,
        project_id: str,
    ):
        super().__init__()
        self.client = client
        self.row = row
        self.image_model = image_model
        self.image_resolution = image_resolution
        self.images_dir = images_dir
        self.project_id = project_id
        self.signals = WorkerSignals()
        self.image_quality = image_quality

    def run(self):
        try:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.images_dir / f"{self.row.ID}.png"

            # --- BASE PROMPTS ---
            prompt = (
                "THIS IS A PRINT COLORING BOOK ILLUSTRATION.\n"
                "BLACK AND WHITE LINE ART ONLY.\n"
                "NO SHADING, NO GRAYS, NO FILLS.\n"
                "PURE WHITE BACKGROUND.\n"
                "LINES MUST BE CLEAN, CONTINUOUS AND PRINT-SAFE.\n\n"

                "DIFFICULTY PARAMETERS (MANDATORY – DO NOT IGNORE):\n"
                f"- Difficulty block: {self.row.Difficulty_block}\n"
                f"- Difficulty score (0–100): {self.row.Difficulty_D}\n"
                f"- Line thickness (L): {self.row.Line_thickness_L}\n"
                f"- White space ratio (W): {self.row.White_space_W}\n"
                f"- Visual complexity (C): {self.row.Complexity_C}\n\n"

                f"{self.row.Prompt_final}\n\n"

                "MANDATORY FINAL CHECK:\n"
                "- Composition MUST touch ALL four edges.\n"
                "- No empty margins.\n"
                "- Foreground elements MUST be cropped by page edges.\n"
            )


            negative = self.row.Prompt_negativo

            negative = f"{self.row.Prompt_negativo}"


            # --- A4.4: Inject rejection reason if regenerating ---
            if (
                self.row.approval_status == "rejected"
                and self.row.rejection_reason.strip()
            ):
                prompt = (
                    f"{prompt}\n\n"
                    "EDITORIAL REVISION FEEDBACK (MANDATORY):\n"
                    "The previous image was rejected for the following reason(s):\n"
                    f"{self.row.rejection_reason}\n\n"
                    "You MUST explicitly correct these issues while strictly respecting:\n"
                    "- all original style rules\n"
                    "- coloring-book constraints\n"
                    "- difficulty parameters\n"
                    "- black-and-white line art only\n"
                )

            generate_image_png(
                client=self.client,
                prompt=prompt,
                negative=negative,
                out_path=out_path,
                image_model=self.image_model,
                image_quality=self.image_quality,
                image_resolution=self.image_resolution,
                project_id=self.project_id,
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
    def start_indeterminate(self, text="🤖 AI is thinking…🤖"):
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

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zoom = 0
        self._zoom_min = -10
        self._zoom_max = 25

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return

        step = 1 if delta > 0 else -1
        new_zoom = self._zoom + step
        if new_zoom < self._zoom_min or new_zoom > self._zoom_max:
            return

        self._zoom = new_zoom
        factor = 1.25 if step > 0 else 0.8
        self.scale(factor, factor)
        event.accept()
        return
    def reset_zoom(self):
        self._zoom = 0
        self.resetTransform()

class ImageReviewDialog(QDialog):

    def __init__(self, parent, row: IllustrationRow, image_path: Path, start_index: int):
        from PySide6.QtWidgets import QScrollArea
        self.parent_window = parent
        self.current_index = start_index
        self.read_only = bool(self.parent_window.editorial_frozen)

              
        super().__init__(parent)   
        self.setWindowFlags(
            Qt.Dialog |
            Qt.Window |
            Qt.WindowMinMaxButtonsHint |
            Qt.WindowCloseButtonHint
        )
        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev.activated.connect(self._go_prev)

        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next.activated.connect(self._go_next)

        self.shortcut_close = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.shortcut_close.activated.connect(self.reject)

        # QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._go_prev)
        # QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._go_next)
        # QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.reject)      # ✅ ESTA LÍNEA ES LA CLAVE
        self.setWindowModality(Qt.WindowModal)   # bloquea SOLO el mainwindow
        self.setWindowTitle(f"Review image {row.ID}")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)


        self.row = row                   # ✅ para _approve/_reject

        
                
        root = QHBoxLayout(self)

        # =========================
        # LEFT: IMAGE (MAX SPACE)
        # =========================
        left = QVBoxLayout()
        root.addLayout(left)
        root.setStretch(0, 1)   # izquierda ocupa todo lo posible
        root.setStretch(1, 0)   # derecha mantiene ancho fijo

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        left.addWidget(self.view, 10)   # ✅ antes estaba en 0

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("← Previous")
        self.btn_next = QPushButton("Next →")
        self.btn_prev.clicked.connect(self._go_prev)
        self.btn_next.clicked.connect(self._go_next)

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        self.lbl_index = QLabel()
        nav.addWidget(self.lbl_index)
        left.addLayout(nav)
        # --- Read-only banner (visible, not just disabled buttons) ---
        self.lbl_readonly = QLabel()
        self.lbl_readonly.setAlignment(Qt.AlignCenter)
        self.lbl_readonly.setStyleSheet("""
            QLabel {
                background-color: #f2f2f2;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 6px 10px;
                font-weight: bold;
            }
        """)

        if self.read_only:
            self.lbl_readonly.setText("🔒 Book closed — review is read-only")
            left.addWidget(self.lbl_readonly)


        # =========================
        # RIGHT: INFO PANEL
        # =========================
        right_container = QWidget()
        right_container.setFixedWidth(600)
        from PySide6.QtWidgets import QSizePolicy

        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # --- Prompt ---
        prompt_box = QGroupBox("Prompt (final + negative)")
        prompt_layout = QVBoxLayout(prompt_box)
        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setReadOnly(True)
        prompt_layout.addWidget(self.txt_prompt)
        right_layout.addWidget(prompt_box)

        # --- Editorial ---
        text_box = QGroupBox("Inspirational text")
        text_layout = QVBoxLayout(text_box)
        self.txt_editorial = QPlainTextEdit()
        self.txt_editorial.setReadOnly(True)
        text_layout.addWidget(self.txt_editorial)
        right_layout.addWidget(text_box)

        # --- Rejection ---
        self.reject_reason = QPlainTextEdit()
        self.reject_reason.setPlaceholderText("Reason for rejection (required)")
        self.reject_reason.setMinimumHeight(140)
        right_layout.addWidget(self.reject_reason)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_approve = QPushButton("Approve")
        self.btn_reject = QPushButton("Reject")
        self.btn_cancel = QPushButton("Close")
        self.lbl_status = QLabel()
        self.lbl_status.setFixedHeight(28)
        self.lbl_status.setMinimumWidth(100)
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("""
            QLabel {
                border-radius: 6px;
                font-weight: bold;
                padding: 4px 10px;
            }
        """)
        btn_layout.addWidget(self.btn_approve)
        btn_layout.addWidget(self.btn_reject)
        btn_layout.addWidget(self.lbl_status)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(btn_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right_container)
      

        root.addWidget(scroll)
        root.setStretch(0, 1)   # left ocupa el resto
        root.setStretch(1, 0)   # right queda fijo
        scroll.setFixedWidth(600)

        if self.read_only:
            self.btn_approve.setEnabled(False)
            self.btn_reject.setEnabled(False)
            self.reject_reason.setReadOnly(True)
            self.reject_reason.setPlaceholderText("Book is closed — read-only review")

        # --- Signals ---
        self.btn_approve.clicked.connect(self._approve)
        self.btn_reject.clicked.connect(self._reject)
        self.btn_cancel.clicked.connect(self.reject)
        self.lbl_index.setText(f"{self.current_index + 1} / {len(self.parent_window.rows)}")
        self._load_current()




    def _approve(self):
        if self.read_only:
            return

        if not self.btn_approve.isEnabled():
            return
        self.row.approval_status = "approved"
        self.row.rejection_reason = ""
        self.parent_window._apply_row_color(self.current_index, "approved")
        self.parent_window.on_save_project(silent=True)
        self.reject_reason.clear()
        self._update_status_badge()
        log_openai_call_v2(
            request_id=str(uuid.uuid4()),
            call_type="image_review",
            model="editorial",
            start_ts=_utc_now_iso(),
            end_ts=_utc_now_iso(),
            status="success",
            phase="review",
            user_action="approve",
            project_id=self.parent_window.current_spec.book_id,
            book_id=self.parent_window.current_spec.book_id,
            row_id=self.row.ID,
            approval_outcome="approved",
        )

        self._go_next()
        if self.current_index == len(self.parent_window.rows) - 1:
          QMessageBox.information(self, "Done", "Last image reviewed.")
        self.update_editorial_status()

  
    def _reject(self):
        if self.read_only:
            return

        if self.row.approval_status == "approved":
            reply = QMessageBox.warning(
                self,
                "Approved image",
                "This image is already APPROVED.\n"
                "Changing its status may affect the final book.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if not self.btn_reject.isEnabled():
            return
        reason = self.reject_reason.toPlainText().strip()
        if not reason:
            QMessageBox.warning(self, "Missing reason", "Please provide a reason for rejection.")
            return
        if self.row.approval_status == "approved":
            reply = QMessageBox.warning(
                self,
                "Already approved",
                "This image is already approved.\nDo you really want to reject it?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        self.row.approval_status = "rejected"
        self.row.rejection_reason = reason
        self.parent_window._apply_row_color(self.current_index, "rejected")
        self.parent_window.on_save_project(silent=True)
        self._update_status_badge()
        log_openai_call_v2(
            request_id=str(uuid.uuid4()),
            call_type="image_review",
            model="editorial",
            start_ts=_utc_now_iso(),
            end_ts=_utc_now_iso(),
            status="success",
            phase="review",
            user_action="reject",
            project_id=self.parent_window.current_spec.book_id,
            book_id=self.parent_window.current_spec.book_id,
            row_id=self.row.ID,
            approval_outcome="rejected",
        )

        self._go_next()
        if self.current_index == len(self.parent_window.rows) - 1:
            QMessageBox.information(self, "Done", "Last image reviewed.")
        self.update_editorial_status()
    def _go_prev(self):
        if self.current_index > 0:
            self.parent_window.on_save_project(silent=True)

            self.current_index -= 1
            self._load_current()

    def _go_next(self):
        if self.current_index < len(self.parent_window.rows) - 1:
            self.parent_window.on_save_project(silent=True)

            self.current_index += 1
            self._load_current()

    def _load_current(self):
        # Always update index and base row first
        self.view.resetTransform()
        row = self.parent_window.rows[self.current_index]
        self.row = row
        self.lbl_index.setText(f"{self.current_index + 1} / {len(self.parent_window.rows)}")

        img_base = self.parent_window.images_dir if self.parent_window.images_dir else OUT_DIR
        img_path = img_base / f"{row.ID}.png"

        # Reset view/scene always
        self.scene.clear()

        # If image missing -> hard disable + clear + placeholder
        if not img_path.exists():
            # Disable actions (hard)
            self.btn_approve.setEnabled(False)
            self.btn_reject.setEnabled(False)

            # Clear text panels (hard)
            self.txt_prompt.setPlainText(f"[{row.ID}] Image not created yet.")
            self.txt_editorial.clear()
            self.reject_reason.clear()

            # Placeholder in the image area
            placeholder = QGraphicsTextItem("Image not created yet")
            placeholder.setDefaultTextColor(QColor("#888888"))
            placeholder.setScale(2.0)
            self.scene.addItem(placeholder)
            self.view.centerOn(placeholder)
            self._update_status_badge()
            return

        # Image exists -> enable actions
        if self.read_only:
            self.btn_approve.setEnabled(False)
            self.btn_reject.setEnabled(False)
            self.reject_reason.setReadOnly(True)
            self.reject_reason.clear()

        else:
            self.btn_approve.setEnabled(True)
            self.btn_reject.setEnabled(True)
            self.reject_reason.setReadOnly(False)

        if row.approval_status == "approved":
            self.btn_approve.setEnabled(False)
            self.btn_reject.setEnabled(False)
            self.reject_reason.setEnabled(False)
        if row.approval_status == "rejected":
            self.btn_reject.setEnabled(False)
        # Load image
        pix = QPixmap(str(img_path))
        self.pixmap_item = self.scene.addPixmap(pix)

        self.view.resetTransform()
        self.view.setSceneRect(self.pixmap_item.boundingRect())

        QTimer.singleShot(0, lambda: (
            self.view.fitInView(
                self.pixmap_item.boundingRect(),
                Qt.KeepAspectRatio
            )
        ))

        # Update right panel texts
        prompt_text = (
            row.Prompt_final
            + "\n\n--- NEGATIVE PROMPT ---\n\n"
            + row.Prompt_negativo
        )

        if row.rejection_reason:
            prompt_text += (
                "\n\n"
                "⚠️ IMPORTANT — PREVIOUS IMAGE REJECTION\n"
                "-------------------------------------\n"
                "A previous version of this image was generated and REJECTED.\n"
                "The following reason was provided by the reviewer and was used "
                "as mandatory feedback to regenerate the current image:\n\n"
                f"{row.rejection_reason}\n"
            )


        self.txt_prompt.setPlainText(prompt_text)

        self.txt_editorial.setPlainText(row.Comentario_editorial)
        self.txt_editorial.setReadOnly(True)
        if self.read_only:
            self.reject_reason.clear()
        else:
            self.reject_reason.setPlainText(row.rejection_reason or "")
        self._update_status_badge()


    def _update_status_badge(self):
        status = self.row.approval_status

        if status == "approved":
            self.lbl_status.setText("APPROVED")
            self.lbl_status.setStyleSheet("""
                QLabel {
                    background-color: #2f6f4e;
                    color: white;
                    border-radius: 6px;
                    font-weight: bold;
                    padding: 4px 10px;
                }
            """)
        elif status == "rejected":
            self.lbl_status.setText("REJECTED")
            self.lbl_status.setStyleSheet("""
                QLabel {
                    background-color: #7a2e2e;
                    color: white;
                    border-radius: 6px;
                    font-weight: bold;
                    padding: 4px 10px;
                }
            """)
        else:
            self.lbl_status.setText("PENDING")
            self.lbl_status.setStyleSheet("""
                QLabel {
                    background-color: #cccccc;
                    color: #333333;
                    border-radius: 6px;
                    font-weight: bold;
                    padding: 4px 10px;
                }
            """)



class BatchImageGenerationDialog(QDialog):
    def __init__(self, parent, rows):
        super().__init__(parent)
        self.parent_window = parent
        self.rows = rows
        self.total = len(rows)
        self.current = 0
        self.visual_index = -1   # índice realmente mostrado en pantalla
        self.stop_requested = False

        self.setWindowTitle("Generating pending images")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(1200, 800)

        root = QHBoxLayout(self)

        # LEFT: image preview + local progress
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignCenter)

        self.lbl_generating = QLabel("Generating image…")
        self.lbl_generating.setAlignment(Qt.AlignCenter)
        self.lbl_generating.setStyleSheet("""
            font-weight: bold;
            color: #dddddd;
            padding: 6px;
        """)

        self.progress_local = QProgressBar()
        self.progress_local.setRange(0, 0)  # indeterminate
        self.progress_local.setFixedHeight(18)

        left_container = QVBoxLayout()
        left_container.addWidget(self.view, 1)
        left_container.addWidget(self.lbl_generating)
        left_container.addWidget(self.progress_local)

        root.addLayout(left_container, 2)


        # RIGHT: info
        right = QVBoxLayout()
        root.addLayout(right, 1)

        self.lbl_progress = QLabel()
        self.lbl_progress.setAlignment(Qt.AlignCenter)
        self.lbl_progress.setStyleSheet("font-size:16px;font-weight:bold;")
        right.addWidget(self.lbl_progress)

        self.txt_info = QPlainTextEdit()
        self.txt_info.setReadOnly(True)
        right.addWidget(self.txt_info)

        self.progress = QProgressBar()
        self.progress.setRange(0, self.total)
        right.addWidget(self.progress)

        self.btn_stop = QPushButton("Stop after current image")
        self.btn_stop.clicked.connect(self._request_stop)
        right.addWidget(self.btn_stop)

        self._update_ui()
        self._start_current()
    def _request_stop(self):
        self.stop_requested = True
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Stopping…")
    def _update_ui(self):
        row = self.rows[self.current]

        self.lbl_progress.setText(
            f"Generating {self.current + 1} / {self.total}"
        )

        self.progress.setValue(self.current)

        # Si aún no hay imagen mostrada, mostramos estado pending
        if self.visual_index < 0:
            info_text = (
                "ID: pending\n"
                "Country: pending\n"
                "Monument: pending\n\n"
                "Waiting for first image..."
            )
        else:
            vrow = self.rows[self.visual_index]
            info_text = (
                f"ID: {vrow.ID}\n"
                f"Country: {vrow.Pais_Region}\n"
                f"Monument: {vrow.Monumento}\n\n"
                f"{vrow.Comentario_editorial}\n\n"
                f"--- FINAL PROMPT ---\n{vrow.Prompt_final}"
            )

        self.txt_info.setPlainText(info_text)

    def _start_current(self):
        self.lbl_generating.show()
        self.progress_local.show()
        row = self.rows[self.current]
        

        image_model = self.parent_window.get_selected_image_model()
        image_resolution = self.parent_window.get_selected_image_resolution()

        worker = GenerateImageWorker(
            self.parent_window.client,
            row,
            image_model,
            self.parent_window.current_spec.image_quality,
            image_resolution,
            self.parent_window.images_dir,
            self.parent_window.current_spec.book_id,
        )


        worker.signals.ok.connect(self._on_image_done)
        worker.signals.err.connect(self._on_error)
        
        self.parent_window.pool.start(worker)
    def _on_image_done(self, out_path: str):
        self.lbl_generating.hide()
        self.progress_local.hide()

        self.scene.clear()
        pix = QPixmap(out_path)
        self.scene.addPixmap(pix)
        self.view.fitInView(
            self.scene.itemsBoundingRect(),
            Qt.KeepAspectRatio,
        )

        row = self.rows[self.current]

        # 🔴 SINCRONIZAR LO QUE SE VE CON LO QUE SE ESCRIBE
        self.visual_index = self.current
        self._update_ui()

        self.parent_window.on_save_project(silent=True)

        self.progress.setValue(self.current + 1)

        for r_idx, r in enumerate(self.parent_window.rows):
            if r.ID == row.ID:
                item = QTableWidgetItem()
                item.setIcon(self.parent_window._icon_ok)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
                self.parent_window.table.setItem(r_idx, 0, item)
                break

    def _on_error(self, msg: str):
        self.lbl_generating.hide()
        self.progress_local.hide()
        QMessageBox.critical(self, "Error", msg)
        self.accept()
def load_all_books_metadata(projects_dir: Path) -> dict:
    books = {}
    for p in projects_dir.iterdir():
        if not p.is_dir():
            continue
        book_json = p / "book.json"
        if not book_json.exists():
            continue
        try:
            data = json.loads(book_json.read_text(encoding="utf-8"))
            spec = data.get("spec", {})
            books[spec.get("book_id")] = {
                "book_id": spec.get("book_id"),
                "book_name": spec.get("book_id"),
                "audience": spec.get("publico"),
                "num_images": spec.get("numero_imagenes"),
            }
        except Exception:
            continue
    return books


def build_book_summary(df, books_meta):
    rows = []

    for book_id, meta in books_meta.items():
        df_b = df[df["book_id"] == book_id]

        img_calls = df_b[df_b["type"] == "generate_image"]
        txt_calls = df_b[df_b["type"] == "generate_book_batch"]

        rows.append({
            "Book": meta["book_name"],
            "Audience": meta["audience"],
            "Planned images": meta["num_images"],
            "Text calls": len(txt_calls),
            "Image calls": len(img_calls),
            "Text cost": round(txt_calls["cost_estimated"].sum(), 4),
            "Image cost": round(img_calls["cost_estimated"].sum(), 4),
            "Total cost": round(df_b["cost_estimated"].sum(), 4),
            "Approved": len(img_calls[img_calls["approval_outcome"] == "approved"]),
            "Rejected": len(img_calls[img_calls["approval_outcome"] == "rejected"]),
            "Errors": len(img_calls[img_calls["status"] == "error"]),
        })

    return rows

class StatsDialog(QDialog):
    def _on_book_selected(self):
        sel = self.books_table.selectionModel().selectedRows()
        if not sel:
            self.selected_book_id = None
            return

        item = self.books_table.item(sel[0].row(), 0)
        self.selected_book_id = item.data(Qt.UserRole)
        self.load_data()



    def __init__(self, parent):
        super().__init__(parent)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        self.setWindowTitle("📊 OpenAI Statistics")
        self.resize(1400, 900)

        self._Figure = Figure
        self._FigureCanvas = FigureCanvas
        # ======================
        # GLOBAL FILTER BAR (TOP)
        # ======================
        filter_bar = QHBoxLayout()

        self.cb_book = QComboBox()
        self.cb_book.addItem("All books")

        self.cb_audience = QComboBox()
        self.cb_audience.addItem("All audiences")

        self.cb_type = QComboBox()
        self.cb_model = QComboBox()
        self.cb_status = QComboBox()
        self.cb_status.addItems(["All status", "success", "error"])

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.load_data)

        filter_bar.addWidget(QLabel("Book"))
        filter_bar.addWidget(self.cb_book)

        filter_bar.addWidget(QLabel("Audience"))
        filter_bar.addWidget(self.cb_audience)

        filter_bar.addWidget(QLabel("Type"))
        filter_bar.addWidget(self.cb_type)

        filter_bar.addWidget(QLabel("Model"))
        filter_bar.addWidget(self.cb_model)

        filter_bar.addWidget(QLabel("Status"))
        filter_bar.addWidget(self.cb_status)

        filter_bar.addStretch()
        filter_bar.addWidget(btn_refresh)

        root = QVBoxLayout(self)
        root.addLayout(filter_bar)
        # ===== TOP: BOOK SUMMARY TABLE =====
        self.books_table = QTableWidget()
        self.books_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.books_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.books_table.verticalHeader().setVisible(False)
        root.addWidget(QLabel("📚 Book summary"))
        root.addWidget(self.books_table)




        # ======================
        # TABLE
        # ======================
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.selected_book_id = None
        self.books_table.itemSelectionChanged.connect(self._on_book_selected)


        self.table.setStyleSheet("""
            QTableWidget { font-size:12px; }
            QHeaderView::section {
                background-color:#2b2b2b;
                color:white;
                font-weight:bold;
                padding:6px;
                border:none;
            }
        """)

        

        # ======================
        # COST COMPARISON AREA
        # ======================
        cost_block = QVBoxLayout()

        self.figure_cost = self._Figure(figsize=(8, 4))
        self.canvas_cost = self._FigureCanvas(self.figure_cost)

        self.lbl_cost_summary = QLabel("Ideal: – € | Real: – € | Δ: – € (–%)")
        self.lbl_cost_summary.setAlignment(Qt.AlignCenter)
        self.lbl_cost_summary.setStyleSheet(
            "font-size:14px;font-weight:bold;padding:6px;"
        )

        cost_block.addWidget(self.canvas_cost, 3)
        cost_block.addWidget(self.lbl_cost_summary, 0)

        root.addLayout(cost_block, 3)

        # ======================
        # TEMPORAL COST (STACKED BARS)
        # ======================
        bottom_block = QVBoxLayout()

        gran_bar = QHBoxLayout()
        gran_bar.addWidget(QLabel("Time aggregation"))

        self.cb_granularity = QComboBox()
        self.cb_granularity.addItems(["Hour", "Day", "Week", "Month"])
        self.cb_granularity.currentIndexChanged.connect(self.load_data)

        gran_bar.addWidget(self.cb_granularity)
        gran_bar.addStretch()

        bottom_block.addLayout(gran_bar)

        self.figure_time = self._Figure(figsize=(10, 4))
        self.canvas_time = self._FigureCanvas(self.figure_time)

        bottom_block.addWidget(self.canvas_time, 4)

        root.addLayout(bottom_block, 4)


        # LOADING LABEL
        self.loading_label = QLabel("⏳ Loading statistics…")
        self.loading_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.loading_label)
                # 🔥 FIX: render inicial
        self.load_data()

        QTimer.singleShot(100, self.load_data)

    # ==================================================
    def load_data(self):
        import pandas as pd
        from matplotlib.ticker import FuncFormatter
        COLOR_IMG = "#1f77b4"   # azul (Images)
        COLOR_TXT = "#ff7f0e"   # naranja (Text)
        if not OPENAI_LOG_CSV_V2.exists():
            QMessageBox.information(self, "No logs", "No log file found.")
            self.loading_label.hide()
            return
        

        df = pd.read_csv(OPENAI_LOG_CSV_V2)

        # ===== BOOK SUMMARY (GLOBAL, NO FILTERS)
        books_meta = load_all_books_metadata(PROJECTS_DIR)
        summary_rows = build_book_summary(df, books_meta)

        headers = list(summary_rows[0].keys()) if summary_rows else []
        self.books_table.setColumnCount(len(headers))
        self.books_table.setHorizontalHeaderLabels(headers)
        self.books_table.setRowCount(len(summary_rows))

        for r, row in enumerate(summary_rows):
            for c, h in enumerate(headers):
                item = QTableWidgetItem(str(row[h]))
                if h == "Book":
                    item.setData(Qt.UserRole, books_meta[row["Book"]]["book_id"])
                self.books_table.setItem(r, c, item)

        # Populate book & audience filters
        self.cb_book.blockSignals(True)
        self.cb_book.clear()
        self.cb_book.addItem("All books")
        for b in books_meta.values():
            self.cb_book.addItem(b["book_id"])
        self.cb_book.blockSignals(False)

        self.cb_audience.blockSignals(True)
        self.cb_audience.clear()
        self.cb_audience.addItem("All audiences")
        for a in sorted(set(b["audience"] for b in books_meta.values())):
            self.cb_audience.addItem(a)
        self.cb_audience.blockSignals(False)

        self.books_table.resizeColumnsToContents()


        self.loading_label.show()
        QApplication.processEvents()
        # ======================
        # APPLY GLOBAL FILTERS
        # ======================
        df_f = df.copy()

        if self.cb_book.currentText() != "All books":
            df_f = df_f[df_f["book_id"] == self.cb_book.currentText()]

        if self.cb_audience.currentText() != "All audiences":
            df_f = df_f[df_f["book_id"].isin([
                k for k, v in books_meta.items()
                if v["audience"] == self.cb_audience.currentText()
            ])]

        if self.cb_type.currentText() != "All types":
            df_f = df_f[df_f["type"] == self.cb_type.currentText()]

        if self.cb_model.currentText() != "All models":
            df_f = df_f[df_f["model"] == self.cb_model.currentText()]

        if self.cb_status.currentText() != "All status":
            df_f = df_f[df_f["status"] == self.cb_status.currentText()]

        # ======================
        # IDEAL vs REAL COST (STACKED)
        # ======================
        self.figure_cost.clear()
        ax = self.figure_cost.add_subplot(111)

        real_text = df_f[df_f["type"] == "generate_book_batch"]["cost_estimated"].sum()
        real_img = df_f[df_f["type"] == "generate_image"]["cost_estimated"].sum()
        real_total = real_text + real_img

        ideal_text = 0.0
        ideal_img = 0.0

        for book_id in df_f["book_id"].unique():
            meta = books_meta.get(book_id)
            if not meta:
                continue
            est = estimate_book_cost(
                pricing=PRICING_DATA,
                text_model=DEFAULT_TEXT_MODEL,
                image_model=DEFAULT_IMAGE_MODEL,
                image_quality="low",
                image_resolution="1024x1536",
                num_images=meta["num_images"],
            )
            ideal_text += est["text_cost"]
            ideal_img += est["image_cost"]

        # Ideal
        ax.barh("Ideal", ideal_img, color=COLOR_IMG, label="Images")
        ax.barh("Ideal", ideal_text, left=ideal_img, color=COLOR_TXT, label="Text")

        # Real
        ax.barh("Real", real_img, color=COLOR_IMG)
        ax.barh("Real", real_text, left=real_img, color=COLOR_TXT)

        ax.set_xlabel("€")
        ax.set_title("Ideal vs Real cost")
        ax.legend()

        delta = real_total - (ideal_text + ideal_img)
        pct = (delta / (ideal_text + ideal_img) * 100) if (ideal_text + ideal_img) else 0

        self.lbl_cost_summary.setText(
            f"Ideal: {(ideal_text+ideal_img):.2f} € | "
            f"Real: {real_total:.2f} € | "
            f"Δ: {delta:+.2f} € ({pct:+.1f}%)"
        )

        self.canvas_cost.draw()


        # ======================
        # TEMPORAL STACKED BARS
        # ======================
        self.figure_time.clear()
        ax = self.figure_time.add_subplot(111)

        df_f["timestamp_start"] = pd.to_datetime(df_f["timestamp_start"], errors="coerce")
        rule = {
            "Hour": "h",    # pandas >= 2.0
            "Day": "D",
            "Week": "W",
            "Month": "ME",  # month-end (estable)
        }[self.cb_granularity.currentText()]


        img = (
            df_f[df_f["type"] == "generate_image"]
            .set_index("timestamp_start")
            .resample(rule)["cost_estimated"]
            .sum()
        )

        txt = (
            df_f[df_f["type"] == "generate_book_batch"]
            .set_index("timestamp_start")
            .resample(rule)["cost_estimated"]
            .sum()
        )

        # 🔥 ALIGN INDICES (CLAVE)
        df_stack = (
            pd.concat([img, txt], axis=1, sort=False)
            .fillna(0.0)
        )
        df_stack.columns = ["image", "text"]
        df_stack = df_stack.sort_index()

        now = pd.Timestamp.utcnow().tz_localize(None)


        if self.cb_granularity.currentText() == "Hour":
            since = now - pd.Timedelta(hours=12)
        elif self.cb_granularity.currentText() == "Day":
            since = now - pd.Timedelta(days=3)
        elif self.cb_granularity.currentText() == "Week":
            since = now - pd.Timedelta(weeks=3)
        else:  # Month
            since = now - pd.DateOffset(months=3)
            
        df_stack.index = df_stack.index.tz_localize(None)

        df_stack = df_stack[df_stack.index >= since]



        ax.bar(df_stack.index, df_stack["image"], label="Images", color=COLOR_IMG)
        ax.bar(
            df_stack.index,
            df_stack["text"],
            bottom=df_stack["image"],
            label="Text",
            color=COLOR_TXT
        )
        
        ax.set_title(f"Cost over time ({self.cb_granularity.currentText()})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.canvas_time.draw()

        if not OPENAI_LOG_CSV_V2.exists():
            QMessageBox.information(self, "No logs", "No log file found.")
            self.loading_label.hide()
            return

        
        # ===== Filters populate
        self.cb_type.clear()
        self.cb_type.addItem("All types")
        self.cb_type.addItems(sorted(df["type"].dropna().unique()))

        self.cb_model.clear()
        self.cb_model.addItem("All models")
        self.cb_model.addItems(sorted(df["model"].dropna().unique()))

        # ===== Apply filters
        if self.cb_type.currentText() != "All types":
            df = df[df["type"] == self.cb_type.currentText()]

        if self.cb_model.currentText() != "All models":
            df = df[df["model"] == self.cb_model.currentText()]

        if self.cb_status.currentText() != "All status":
            df = df[df["status"] == self.cb_status.currentText()]

        # ======================
        # TABLE
        # ======================
        
        self.table.resizeColumnsToContents()




class MainWindow(QMainWindow):
    def _lock_ui(self):
        self.centralWidget().setEnabled(False)

    def _unlock_ui(self):
        self.centralWidget().setEnabled(True)
        self.overlay.stop()
    def get_selected_image_resolution(self) -> str:
        return self.cb_image_resolution.currentText().strip()

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

    def update_text_price_label(self):
        model = self.cb_text_model.currentText()
        model_cfg = self.pricing.get("models", {}).get(model)

        if not model_cfg or model_cfg.get("type") != "text":
            self.lbl_text_price.setText("N/A")
            return

        in_p = model_cfg.get("input_token")
        out_p = model_cfg.get("output_token")

        if in_p is None or out_p is None:
            self.lbl_text_price.setText("N/A")
            return

        self.lbl_text_price.setText(
            f"In: {in_p:.4f} € / 1K | Out: {out_p:.4f} € / 1K"
        )


    def update_image_price_label(self):
        model = self.cb_image_model.currentText()
        quality = self.cb_image_quality.currentText()
        resolution = self.cb_image_resolution.currentText()

        model_cfg = self.pricing.get("models", {}).get(model)

        if not model_cfg or model_cfg.get("type") != "image":
            self.lbl_image_price.setText("N/A")
            return

        prices = model_cfg.get("prices", {})
        price = prices.get(quality, {}).get(resolution)

        if price is None:
            self.lbl_image_price.setText("N/A")
        else:
            self.lbl_image_price.setText(f"{price:.4f} € / imagen")

    def update_estimated_book_cost(self):
        try:
            num_images = self.sp_num.value()
            est = estimate_book_cost(
                pricing=self.pricing,
                text_model=self.get_selected_text_model(),
                image_model=self.get_selected_image_model(),
                image_quality=self.cb_image_quality.currentText(),
                image_resolution=self.get_selected_image_resolution(),
                num_images=num_images,
            )

            self.lbl_estimated_cost.setText(
                f"Estimated cost (no retries): "
                f"{est['total_cost']} € "
                f"(Text: {est['text_cost']} €, Images: {est['image_cost']} €)"
            )
        except Exception:
            self.lbl_estimated_cost.setText("Estimated cost: N/A")
    def on_close_book_editorial(self):
        if self.project_dir is None or self.images_dir is None or not self.rows:
            QMessageBox.warning(self, "Close book", "Open a project with images first.")
            return

        if getattr(self, "editorial_frozen", False):
            QMessageBox.information(self, "Close book", "This book is already frozen.")
            return

        # 1) Validar que TODAS las imágenes existen
        missing_ids = []
        for r in self.rows:
            if not (self.images_dir / f"{r.ID}.png").exists():
                missing_ids.append(r.ID)

        if missing_ids:
            # no listamos todas si son muchas
            sample = ", ".join(missing_ids[:10])
            more = "" if len(missing_ids) <= 10 else f" (+{len(missing_ids)-10} more)"
            QMessageBox.critical(
                self,
                "Cannot close book",
                f"Missing images: {len(missing_ids)}\n\n"
                f"Examples: {sample}{more}\n\n"
                "Generate all missing images before closing the book.",
            )
            return

        # 2) Si hay pending/rejected → forzar a approved
        not_approved = [r for r in self.rows if r.approval_status != "approved"]

        if not_approved:
            reply = QMessageBox.warning(
                self,
                "Unapproved images",
                f"All images exist, but {len(not_approved)} image(s) are not approved.\n\n"
                "They will be forced to APPROVED and the book will be CLOSED.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

            for r in not_approved:
                r.approval_status = "approved"
                # cierre editorial: limpia motivos para no arrastrar “rechazos” en un libro cerrado
                r.rejection_reason = f"[AUTO-APPROVED ON FREEZE]\n{r.rejection_reason}".strip()


        # 3) Freeze
        self.editorial_frozen = True
        self.set_book_inputs_enabled(False)

        if isinstance(getattr(self, "book_data", None), dict):
            self.book_data["editorial_frozen"] = True

        # 4) Refrescar colores (y contadores si los tienes)
        for i, r in enumerate(self.rows):
            self._apply_row_color(i, r.approval_status)

        if hasattr(self, "update_editorial_status"):
            self.update_editorial_status()

        # 5) Guardar
        self.on_save_project(silent=False)

        QMessageBox.information(
            self,
            "Book closed",
            "Book closed successfully.\n\n"
            "Image generation is now disabled.",
        )
        self.action_generar_imagen.setEnabled(False)
        self.action_generar_imagenes_pendientes.setEnabled(False)
        self.action_generar_libro.setEnabled(False)
        self.action_close_book.setEnabled(False)
        self._refresh_action_states()

    def on_generate_pending_images(self):
        if getattr(self, "editorial_frozen", False):
            QMessageBox.warning(self, "Book closed", "This book is editorially frozen.")
            return



        if not self.images_dir:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return

        pending = [
            row for row in self.rows
            if (
                row.approval_status == "pending"
                or (
                    row.approval_status == "rejected"
                    and row.rejection_reason.strip()
                )
            )
        ]


        if not pending:
            QMessageBox.information(
                self,
                "Nothing to do",
                "There are no pending images to generate.",
            )
            return

        dlg = BatchImageGenerationDialog(self, pending)
        dlg.exec()

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
    def on_open_stats(self):
        dlg = StatsDialog(self)
        dlg.exec()

    def on_export_xls(self):
        if not getattr(self, "editorial_frozen", False):
            QMessageBox.warning(
                self,
                "Export not allowed",
                "The book must be CLOSED before exporting to XLS.",
            )
            return

        from re import sub

        raw_title = self.ed_tema.toPlainText().strip()
        safe_title = sub(r"[^\w\s-]", "", raw_title).strip().replace(" ", "_")
        default_name = f"{safe_title or 'canva_import'}.xlsx"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export XLS for Canva",
            default_name,
            "Excel Files (*.xlsx)",
        )

        if not path:
            return

        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "Canva Import"

        # Header EXACTO requerido por Canva
        ws.append([
            "ID",
            "Monumento",
            "Pais/region",
            "Main Foco",
            "Comentario Editorial",
        ])

        for r in self.rows:
            ws.append([
                r.ID,
                r.Monumento,
                r.Pais_Region,
                r.Main_foco,
                r.Comentario_editorial,
            ])

        wb.save(path)

        QMessageBox.information(
            self,
            "Export completed",
            f"XLS exported successfully:\n{path}",
        )


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
    def update_editorial_status(self):
        approved = sum(1 for r in self.rows if r.approval_status == "approved")
        rejected = sum(1 for r in self.rows if r.approval_status == "rejected")
        pending = len(self.rows) - approved - rejected
        self.action_export_xls.setEnabled(self.editorial_frozen)

        self.lbl_editorial_status.setText(
            f"Approved: {approved} | Pending: {pending} | Rejected: {rejected}"
        )
        self.btn_action_generate_book.setEnabled(not self.editorial_frozen)
        self.btn_action_generate_images.setEnabled(not self.editorial_frozen)
        self._refresh_action_states()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Moxalia – Coloring Book Studio")
        # --- Pricing data ---
        self.pricing = PRICING_DATA
        self.resize(1400, 900)              # tamaño inicial razonable
        self.setMinimumSize(1280, 800)      # 👈 clave para pantallas 14"
        # --- Editorial status indicators ---
       
        self.client = OpenAI()
        self.pool = QThreadPool.globalInstance()
        self.overlay = BlockingOverlay(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.hide()
        self.req_imagenes = load_requirement_file(REQ_IMAGENES_PATH)
        self.req_texto = load_requirement_file(REQ_TEXTO_PATH)
        self.project_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None
        self.book_data: Dict[str, Any] = {}
        self.editorial_frozen: bool = False

        self.rows: List[IllustrationRow] = []
        self.current_spec: Optional[BookSpec] = None

        # New: Available model lists (can be extended at will)
        self.available_text_models = [
            "gpt-4.1-mini",
            "gpt-4",
            "gpt-3.5-turbo-1106",
        ]
        self.available_image_models = [
            "dall-e-3",
            "gpt-image-1.5",
            "gpt-image-1",
            "gpt-image-1-mini",
        ]

        # ----- MENU BAR -----
        menubar = self.menuBar()
        archivo_menu = menubar.addMenu("Archivo")
        acciones_menu = menubar.addMenu("Acciones")
        stats_menu = menubar.addMenu("Estadísticas")

        self.action_stats = QAction("OpenAI Statistics", self)
        self.action_stats.triggered.connect(self.on_open_stats)
        stats_menu.addAction(self.action_stats)


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

        self.action_generar_imagenes_pendientes = QAction(
            "Generar imágenes pendientes", self
        )
        self.action_generar_imagenes_pendientes.setToolTip("Generate all missing images for rows that have no image yet.")

        self.action_generar_imagenes_pendientes.setEnabled(False)
        self.action_generar_imagenes_pendientes.triggered.connect(
            self.on_generate_pending_images
        )
        acciones_menu.addAction(self.action_generar_imagenes_pendientes)

        self.action_close_book = QAction("Close book (editorial freeze)", self)
        self.action_close_book.setToolTip("Freeze the book: lock generation and allow read-only review. Export is enabled after freezing.")
        self.action_close_book.triggered.connect(self.on_close_book_editorial)
        acciones_menu.addAction(self.action_close_book)

        self.action_export_xls = QAction("Export XLS for Canva", self)
        self.action_export_xls.setToolTip("Export is only enabled after the book is closed (frozen).")
        self.action_export_xls.triggered.connect(self.on_export_xls)
        acciones_menu.addAction(self.action_export_xls)
        self.action_export_xls.setEnabled(False)


        # ----- CENTRAL UI -----
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QVBoxLayout()
        main.addLayout(left, 3)
        right = QVBoxLayout()
        main.addLayout(right, 1)

        # --- Editorial status indicators (A4.3) ---
        self.lbl_editorial_status = QLabel("Approved: 0 | Pending: 0 | Rejected: 0")
        self.lbl_editorial_status.setAlignment(Qt.AlignLeft)
        self.lbl_editorial_status.setStyleSheet(
            "font-weight: bold; padding: 6px;"
        )
        left.addWidget(self.lbl_editorial_status)
        # =========================
        # TOP AREA — 3 COLUMNS
        # =========================
        top_area = QHBoxLayout()
        left.addLayout(top_area)

        # -------- COLUMN 1: ACTIONS --------
        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)

        btn_save = QPushButton("Save project")
        btn_save.clicked.connect(self.on_save_project)

        self.btn_action_generate_book = QPushButton("Generate book")
        self.btn_action_generate_book.clicked.connect(self.on_generate_book)

        self.btn_action_generate_images = QPushButton("Generate pending images")
        self.btn_action_generate_images.setToolTip("Generate all missing images for rows that do not have an image yet.")
        self.btn_action_generate_images.clicked.connect(self.on_generate_pending_images)

        btn_close_book = QPushButton("Close book")
        btn_close_book.setToolTip("Freeze the book: lock generation and allow read-only review. Export becomes available.")
        btn_close_book.clicked.connect(self.on_close_book_editorial)

        actions_layout.addWidget(btn_save)
        actions_layout.addWidget(self.btn_action_generate_book)
        actions_layout.addWidget(self.btn_action_generate_images)
        actions_layout.addWidget(btn_close_book)
        actions_layout.addStretch()

        top_area.addWidget(actions_box, 1)

        # -------- COLUMN 2: PROJECT --------
        self.lbl_estimated_cost = QLabel("Estimated cost: N/A")
        self.lbl_estimated_cost.setStyleSheet("font-weight: bold; color: #d0d0d0;")
        project_box = QGroupBox("Project")
        project_layout = QVBoxLayout(project_box)

        self.lbl_project = QLabel("No book loaded")
        project_layout.addWidget(self.lbl_project)
        project_layout.addWidget(self.lbl_estimated_cost)
        top_area.addWidget(project_box, 2)



        # -------- COLUMN 3: EDITORIAL REQUIREMENTS --------
        req_box = QGroupBox("Editorial requirements")
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
        req_layout.addSpacing(12)
        req_layout.addWidget(self.lbl_req_txt)
        req_layout.addStretch()

        top_area.addWidget(req_box, 2)
        # -------- BOOK INPUTS --------
        form_box = QGroupBox("Book inputs")

        # -------- BOOK INPUTS --------
        form_box = QGroupBox("Book inputs")
        left.addWidget(form_box)

        form_layout = QHBoxLayout(form_box)


        form_left = QFormLayout()
        form_right = QFormLayout()

        form_layout.addLayout(form_left)
        form_layout.addLayout(form_right)

        self.ed_book_id = QLineEdit()
        self.cb_publico = QComboBox()
        self.cb_publico.addItems(["Adultos", "Niños"])
        self.ed_tema = QPlainTextEdit()
        self.ed_tema.setPlaceholderText("Tema del libro")
        self.ed_tema.setFixedHeight(80)
        self.ed_tema.setLineWrapMode(QPlainTextEdit.WidgetWidth)

        self.ed_alcance = QLineEdit()
        self.ed_tipo = QLineEdit("Monumentos y paisajes")
        self.sp_num = QSpinBox()
        self.sp_num.setRange(1, 500)
        self.sp_num.setValue(50)

        self.ed_tamano = QLineEdit("8.5x11")
        self.ed_aspect = QLineEdit("22:17")

        self.cb_dif_ini = QComboBox()
        self.cb_dif_ini.addItems(["Bajo", "Medio", "Alto", "Extremo"])
        self.cb_dif_fin = QComboBox()
        self.cb_dif_fin.addItems(["Bajo", "Medio", "Alto", "Extremo"])
        self.cb_dif_fin.setCurrentText("Alto")

        self.cb_idioma = QComboBox()
        self.cb_idioma.addItems(["es", "en", "it", "fr", "de"])
        self.cb_idioma.setCurrentText("es")
        # --- Pricing info (read-only) ---
        self.lbl_text_price = QLabel("-")
        self.lbl_image_price = QLabel("-")

        self.lbl_text_price.setStyleSheet("color: #9da9b5;")
        self.lbl_image_price.setStyleSheet("color: #9da9b5;")

        form_right.addRow("Precio texto (€/M tokens)", self.lbl_text_price)
        form_right.addRow("Precio imagen (€/imagen)", self.lbl_image_price)



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
        # Las resoluciones se cargarán dinámicamente según el modelo seleccionado    
        
        self.cb_image_model = QComboBox()
        for m in self.available_image_models:
            self.cb_image_model.addItem(m)
        idx_image = self.cb_image_model.findText(DEFAULT_IMAGE_MODEL)
        if idx_image != -1:
            self.cb_image_model.setCurrentIndex(idx_image)
        self.cb_image_model.currentTextChanged.connect(
            self.on_image_model_changed
        )
        self.cb_image_quality = QComboBox()
        self.cb_image_quality.addItems(["low", "medium", "high"])
        self.cb_image_quality.setCurrentText("low")
        form_right.addRow("Image model", self.cb_image_model)       
        form_right.addRow("Image quality", self.cb_image_quality)
        form_right.addRow("Image resolution", self.cb_image_resolution)
        self.on_image_model_changed(self.cb_image_model.currentText())

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
        # --- Estimated cost recalculation ---
        self.sp_num.valueChanged.connect(self.update_estimated_book_cost)
        self.cb_text_model.currentTextChanged.connect(self.update_estimated_book_cost)
        self.cb_image_model.currentTextChanged.connect(self.update_estimated_book_cost)
        self.cb_image_quality.currentTextChanged.connect(self.update_estimated_book_cost)
        self.cb_image_resolution.currentTextChanged.connect(self.update_estimated_book_cost)

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

        # Update pricing labels on change
        self.cb_text_model.currentTextChanged.connect(
            lambda _: self.update_text_price_label()
        )

        self.cb_image_model.currentTextChanged.connect(
            lambda _: self.update_image_price_label()
        )
        self.cb_image_quality.currentTextChanged.connect(
            lambda _: self.update_image_price_label()
        )
        self.cb_image_resolution.currentTextChanged.connect(
            lambda _: self.update_image_price_label()
        )

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
        self.update_text_price_label()
        self.update_image_price_label()
        self.update_estimated_book_cost()


        self.validate_form()
    def on_image_model_changed(self, model: str):
        model = model.strip()

        caps = IMAGE_MODEL_CAPABILITIES.get(model)
        if not caps:
            return

        # --- Quality ---
        self.cb_image_quality.blockSignals(True)
        self.cb_image_quality.clear()
        self.cb_image_quality.addItems(caps["qualities"])
        self.cb_image_quality.setCurrentText(caps["default_quality"])
        self.cb_image_quality.blockSignals(False)

        # --- Resolution ---
        self.cb_image_resolution.blockSignals(True)
        self.cb_image_resolution.clear()
        self.cb_image_resolution.addItems(caps["sizes"])
        self.cb_image_resolution.setCurrentText(caps["default_size"])
        self.cb_image_resolution.blockSignals(False)
    def _refresh_action_states(self):
        has_rows = bool(self.rows)
        has_project = self.project_dir is not None
        frozen = bool(getattr(self, "editorial_frozen", False))

        # Menu
        self.action_generar_libro.setEnabled(has_project and not frozen)
        self.action_generar_imagen.setEnabled(has_rows and not frozen and bool(self.table.selectionModel().selectedRows()))
        self.action_generar_imagenes_pendientes.setEnabled(has_rows and has_project and not frozen)
        self.action_close_book.setEnabled(has_rows and has_project and not frozen)
        self.action_export_xls.setEnabled(frozen)

        # Actions panel buttons
        self.btn_action_generate_book.setEnabled(has_project and not frozen)
        self.btn_action_generate_images.setEnabled(has_rows and has_project and not frozen)

    # ---- Menu action enablement helpers ----
    def validate_form(self):
        has_project = self.project_dir is not None
        ok = (
            self.ed_book_id.text().strip() != ""
            and self.ed_tema.toPlainText().strip() != ""
            and self.ed_alcance.text().strip() != ""
        )
        self.btn_generate_book.setEnabled(ok)
        # Sync menu action enable state:
        if hasattr(self, "action_generar_libro"):
            self.action_generar_libro.setEnabled(ok)
        
        if hasattr(self, "action_generar_imagenes_pendientes"):
            self.action_generar_imagenes_pendientes.setEnabled(
                self.project_dir is not None and bool(self.rows)
            )

        self._refresh_action_states()
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
            self.lbl_image_preview.setPixmap(QPixmap())
            self.lbl_image_preview.setText("Select a row to preview the image")
            self.lbl_image_preview.setStyleSheet("""
                QLabel {
                    background-color: #e0e0e0;
                    color: #555;
                    border: 1px dashed #aaa;
                }
            """)
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
            self.lbl_image_preview.setText("Image not generated yet")
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
            self.book_data = book_data
            self.editorial_frozen = bool(book_data.get("editorial_frozen", False))
            self.set_book_inputs_enabled(not self.editorial_frozen)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer book.json:\n{e}")
            return

        # Restore book form fields
        try:
            # Handle missing keys gracefully if needed
            self.ed_book_id.setText(spec.get("book_id", ""))
            self.cb_publico.setCurrentText(spec.get("publico", "Adultos"))
            self.ed_tema.setPlainText(spec.get("tema", ""))   # ← AQUÍ
            self.ed_alcance.setText(spec.get("alcance_geografico", ""))
            self.ed_tipo.setText(spec.get("tipo_imagenes", ""))
            self.sp_num.setValue(spec.get("numero_imagenes", 0))
            self.ed_tamano.setText(spec.get("tamano_libro", "8.5x11"))
            self.ed_aspect.setText(spec.get("aspect_ratio", "22:17"))
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

            image_settings = book_data.get("image_settings", {})

            image_model = image_settings.get("image_model", DEFAULT_IMAGE_MODEL)
            if isinstance(image_model, str):
                image_model = image_model.strip()


            # 🔴 NORMALIZACIÓN CRÍTICA
            if isinstance(image_model, str):
                image_model = image_model.strip().lower()

            idx = self.cb_image_model.findText(image_model)
            if idx != -1:
                self.cb_image_model.setCurrentIndex(idx)
            else:
                self.cb_image_model.setCurrentIndex(0)



            image_resolution = spec.get("image_resolution", "Working (1024 x 1536)")
            idx = self.cb_image_resolution.findText(image_resolution)
            if idx != -1:
                self.cb_image_resolution.setCurrentIndex(idx)
            else:
                self.cb_image_resolution.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error restaurando campos de book.json:\n{e}")
            return
        image_quality = spec.get("image_quality", "low")
        idx = self.cb_image_quality.findText(image_quality)
        if idx != -1:
            self.cb_image_quality.setCurrentIndex(idx)
        else:
            self.cb_image_quality.setCurrentIndex(0)

        # 🔴 FIX CRÍTICO: restaurar current_spec
        try:
            self.current_spec = BookSpec(**spec)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Invalid book spec in book.json:\n{e}"
            )
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
            img_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

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
        self.update_editorial_status()
        self.update_estimated_book_cost()
        self._refresh_action_states()
    def on_save_project(self, *, silent: bool = False):
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
                    "editorial_frozen": bool(getattr(self, "editorial_frozen", False)),
                    "spec": spec.model_dump(),
                    "image_settings": {
                        "image_model": self.get_selected_image_model(),
                        "image_resolution": self.get_selected_image_resolution(),
                        "image_quality": self.cb_image_quality.currentText(),
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

        if not silent:
            QMessageBox.information(self, "Saved", "Project saved successfully.")
        self.validate_form()
        self._refresh_action_states()


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
        self.editorial_frozen = False
        self.set_book_inputs_enabled(True)
        self._refresh_action_states()

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
    def set_book_inputs_enabled(self, enabled: bool) -> None:
        widgets = [
            self.ed_book_id,
            self.cb_publico,
            self.ed_tema,
            self.ed_alcance,
            self.ed_tipo,
            self.sp_num,
            self.ed_tamano,
            self.ed_aspect,
            self.cb_dif_ini,
            self.cb_dif_fin,
            self.cb_idioma,
            self.cb_text_model,
            self.cb_image_resolution,
            self.cb_image_model,
        ]
        for w in widgets:
            w.setEnabled(enabled)




    def get_spec(self) -> BookSpec:
        return BookSpec(
            book_id=self.ed_book_id.text().strip(),
            publico=self.cb_publico.currentText(),
            tema=self.ed_tema.toPlainText().strip(),
            alcance_geografico=self.ed_alcance.text().strip(),
            tipo_imagenes=self.ed_tipo.text().strip(),
            numero_imagenes=int(self.sp_num.value()),
            tamano_libro=self.ed_tamano.text().strip(),
            aspect_ratio=self.ed_aspect.text().strip(),
            dificultad_inicial=self.cb_dif_ini.currentText(),
            dificultad_final=self.cb_dif_fin.currentText(),
            idioma_texto=self.cb_idioma.currentText(),
            image_quality=self.cb_image_quality.currentText(),
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

            # ✅ F2.1 — Force save BEFORE generating
            self.on_save_project(silent=True)

        except ValidationError as e:
            QMessageBox.critical(self, "Invalid inputs", str(e))
            return

        # Validaciones duras
        if self.project_dir is None:
            QMessageBox.critical(
                self,
                "No book loaded",
                "Create or open a book first.",
            )
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
        self.overlay.start_indeterminate("🤖 AI is generating the book…🤖")
        self._lock_ui()

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
        self._unlock_ui()

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
                icon_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

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
        self.update_editorial_status()  
        self._refresh_action_states()

    def on_image_generated(self, out_path: str):
        self.overlay.stop()
        self.btn_generate_image.setEnabled(True)
        self.lbl_image_status.setText(f"Saved: {out_path}")
        self._unlock_ui()

        # localizar la fila correspondiente y marcarla pending
        for idx, r in enumerate(self.rows):
            if out_path.endswith(f"{r.ID}.png"):
                r.approval_status = "pending"
                r.rejection_reason = ""
                self._apply_row_color(idx, "pending")
                break
        self.update_editorial_status()

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
        self.update_editorial_status()


    def on_generate_image(self):
        if getattr(self, "editorial_frozen", False):
            QMessageBox.warning(self, "Book closed", "This book is editorially frozen.")
            return

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
        image_resolution = self.get_selected_image_resolution()
        self._lock_ui()

        row_idx = self.table.selectionModel().selectedRows()[0].row()
        row = self.rows[row_idx]

        self.overlay.start_indeterminate(
            f"🤖Generating image {row.ID}…🤖"
        )

        worker = GenerateImageWorker(
            self.client,
            row,
            image_model,
            self.current_spec.image_quality,
            image_resolution,
            self.images_dir,
            self.current_spec.book_id,
        )


        worker.signals.ok.connect(self.on_image_generated)
        worker.signals.err.connect(self.on_worker_error)
        self.pool.start(worker)

    def on_worker_error(self, msg: str):
        self.overlay.stop()
        self._unlock_ui()
        self.btn_generate_book.setEnabled(True)
        self.btn_generate_image.setEnabled(True)
        
        QMessageBox.critical(self, "Error", msg)
    def on_table_cell_clicked(self, row_idx, col_idx):
        return

    def on_table_double_clicked(self, row_idx, col_idx):
        # Allow review in read-only mode when frozen
        # (ImageReviewDialog already handles read_only based on parent_window.editorial_frozen)


        if not (0 <= row_idx < len(self.rows)):
            return

        row = self.rows[row_idx]
        img_base = self.images_dir if self.images_dir else OUT_DIR
        img_path = img_base / f"{row.ID}.png"

        # Si no hay imagen, preguntar si generar (permitido incluso si está approved)
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

        # Si hay imagen, SIEMPRE abrir review (aunque sea approved)
        try:
            dlg = ImageReviewDialog(self, row, img_path, row_idx)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Image review error",
                f"Failed to open image review window:\n{e}"
            )


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
