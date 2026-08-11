"""Local API for TR-Hash detector inference and isolated training jobs."""

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from safetensors.torch import load_file

from .config import TRHashDetectorConfig
from .data import _letterbox, _normalize_image
from .model import TRHashObjectDetector


def resolve_device(override: Optional[str]) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelRuntime:
    """Thread-safe detector that can atomically hot-reload a checkpoint."""

    def __init__(self, checkpoint: Path, device: torch.device):
        self.device = device
        self.lock = threading.RLock()
        self.model: Optional[TRHashObjectDetector] = None
        self.config: Optional[TRHashDetectorConfig] = None
        self.validation: Dict[str, float] = {}
        self.checkpoint = Path()
        self.loaded_at = 0.0
        self.load(checkpoint)

    def load(self, checkpoint: Path) -> None:
        checkpoint = Path(checkpoint).resolve()
        config = TRHashDetectorConfig.from_dict(
            json.loads((checkpoint / "config.json").read_text())
        )
        model = TRHashObjectDetector(config)
        model.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
        model.to(self.device).eval()
        validation_path = checkpoint / "validation.json"
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        with torch.inference_mode():
            model(torch.zeros(1, 3, config.image_size, config.image_size, device=self.device))
        with self.lock:
            self.model = model
            self.config = config
            self.validation = validation
            self.checkpoint = checkpoint
            self.loaded_at = time.time()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        *,
        confidence: float,
        iou_threshold: float,
    ) -> List[Dict[str, object]]:
        with self.lock:
            if self.model is None or self.config is None:
                raise RuntimeError("model is not loaded")
            original_w, original_h = image.size
            image, _ = _letterbox(image.convert("RGB"), torch.empty(0, 5), self.config.image_size)
            pixels = _normalize_image(image).unsqueeze(0).to(self.device)
            detection = self.model.predict(
                pixels,
                confidence_threshold=confidence,
                iou_threshold=iou_threshold,
                postprocess_on_cpu=self.device.type == "mps",
            )[0]
            scale = min(
                self.config.image_size / original_w,
                self.config.image_size / original_h,
            )
            resized_w = round(original_w * scale)
            resized_h = round(original_h * scale)
            left = (self.config.image_size - resized_w) // 2
            top = (self.config.image_size - resized_h) // 2
            results = []
            for box, score, label in zip(
                detection["boxes"], detection["scores"], detection["labels"]
            ):
                canvas_box = box.cpu() * self.config.image_size
                x1 = float(((canvas_box[0] - left) / scale).clamp(0, original_w))
                y1 = float(((canvas_box[1] - top) / scale).clamp(0, original_h))
                x2 = float(((canvas_box[2] - left) / scale).clamp(0, original_w))
                y2 = float(((canvas_box[3] - top) / scale).clamp(0, original_h))
                results.append(
                    {
                        "box_xyxy": [x1, y1, x2, y2],
                        "box_normalized": [
                            x1 / original_w,
                            y1 / original_h,
                            x2 / original_w,
                            y2 / original_h,
                        ],
                        "score": float(score),
                        "label": int(label),
                    }
                )
            return results

    def info(self) -> Dict[str, object]:
        with self.lock:
            if self.model is None or self.config is None:
                return {"loaded": False}
            return {
                "loaded": True,
                "checkpoint": str(self.checkpoint),
                "loaded_at": self.loaded_at,
                "device": str(self.device),
                "parameters": self.model.num_parameters(),
                "config": self.config.to_dict(),
                "validation": self.validation,
                "recommended_confidence": self.validation.get("best_confidence", 0.25),
            }


@dataclass
class JobRecord:
    id: str
    status: str
    command: List[str]
    output_dir: str
    log_path: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    return_code: Optional[int] = None


class TrainingJobManager:
    """One-at-a-time training queue with persisted metadata and logs."""

    def __init__(self, root: Path, python_executable: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable
        self.lock = threading.RLock()
        self.jobs: Dict[str, JobRecord] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self._restore_jobs()

    def _restore_jobs(self) -> None:
        """Restore job metadata after a service restart.

        A subprocess cannot be reattached safely, so jobs that were active when
        the server stopped are explicitly marked as interrupted.
        """
        for metadata_path in self.root.glob("*/job.json"):
            try:
                record = JobRecord(**json.loads(metadata_path.read_text()))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if record.status in {"queued", "running", "cancelling"}:
                record.status = "interrupted"
                record.finished_at = time.time()
                self._persist(record)
            self.jobs[record.id] = record

    def _persist(self, record: JobRecord) -> None:
        job_dir = self.root / record.id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(asdict(record), indent=2) + "\n")

    def start(self, request, device: str) -> JobRecord:
        with self.lock:
            if any(job.status in {"queued", "running"} for job in self.jobs.values()):
                raise RuntimeError("a training job is already running")
            job_id = uuid.uuid4().hex
            job_dir = self.root / job_id
            output_dir = job_dir / "output"
            log_path = job_dir / "training.log"
            command = [
                self.python_executable,
                "-u",
                "-m",
                "complexity.generative.detection.training",
                "--output",
                str(output_dir),
                "--device",
                device,
                "--image-size",
                str(request.image_size),
                "--patch-size",
                str(request.patch_size),
                "--epochs",
                str(request.epochs),
                "--batch-size",
                str(request.batch_size),
                "--validation-samples",
                str(request.validation_samples),
                "--validation-fraction",
                str(request.validation_fraction),
                "--vision-hidden-size",
                str(request.vision_hidden_size),
                "--vision-layers",
                str(request.vision_layers),
                "--vision-heads",
                str(request.vision_heads),
                "--vision-expert-width",
                str(request.vision_expert_width),
                "--assignment-top-k",
                str(request.assignment_top_k),
                "--reg-max",
                str(request.reg_max),
                "--head-hidden-size",
                str(request.head_hidden_size),
                "--dfl-loss-weight",
                str(request.dfl_loss_weight),
                "--quality-focal-beta",
                str(request.quality_focal_beta),
                "--augmentation",
                request.augmentation,
                "--momentum",
                str(request.momentum),
                "--weight-decay",
                str(request.weight_decay),
                "--lr",
                str(request.learning_rate),
                "--expert-lr-multiplier",
                str(request.expert_lr_multiplier),
                "--seed",
                str(request.seed),
            ]
            if not request.p2_head:
                command.append("--no-p2-head")
            if not request.stal:
                command.append("--no-stal")
            if not request.progressive_loss:
                command.append("--no-progressive-loss")
            if request.backbone_checkpoint and request.detector_checkpoint:
                raise ValueError(
                    "backbone_checkpoint and detector_checkpoint are mutually exclusive"
                )
            if request.class_map and not request.detector_checkpoint:
                raise ValueError("class_map requires detector_checkpoint")
            if request.backbone_checkpoint:
                command.extend(("--backbone-checkpoint", request.backbone_checkpoint))
            if request.detector_checkpoint:
                command.extend(("--detector-checkpoint", request.detector_checkpoint))
            if request.class_map:
                command.extend(("--class-map", request.class_map))
            if request.dataset == "synthetic":
                command.extend(("--synthetic-samples", str(request.synthetic_samples)))
            else:
                for option, value in (
                    ("--yolo-images", request.yolo_images),
                    ("--yolo-labels", request.yolo_labels),
                ):
                    if not value:
                        raise ValueError(f"{option} is required for YOLO datasets")
                    command.extend((option, value))
                for option, value in (
                    ("--validation-yolo-images", request.validation_yolo_images),
                    ("--validation-yolo-labels", request.validation_yolo_labels),
                ):
                    if value:
                        command.extend((option, value))
            record = JobRecord(
                id=job_id,
                status="queued",
                command=command,
                output_dir=str(output_dir),
                log_path=str(log_path),
                created_at=time.time(),
            )
            self.jobs[job_id] = record
            self._persist(record)
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key != "PYTORCH_ENABLE_MPS_FALLBACK"
                },
            )
            record.status = "running"
            record.started_at = time.time()
            self.processes[job_id] = process
            self._persist(record)
            threading.Thread(
                target=self._wait,
                args=(job_id, process, log_handle),
                daemon=True,
            ).start()
            return record

    def _wait(self, job_id: str, process: subprocess.Popen, log_handle) -> None:
        return_code = process.wait()
        log_handle.close()
        with self.lock:
            record = self.jobs[job_id]
            record.return_code = return_code
            record.finished_at = time.time()
            record.status = (
                "cancelled"
                if record.status == "cancelling"
                else ("completed" if return_code == 0 else "failed")
            )
            self.processes.pop(job_id, None)
            self._persist(record)

    def get(self, job_id: str) -> JobRecord:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def cancel(self, job_id: str) -> JobRecord:
        with self.lock:
            record = self.get(job_id)
            process = self.processes.get(job_id)
            if process is None or record.status != "running":
                raise RuntimeError("job is not running")
            record.status = "cancelling"
            process.terminate()
            self._persist(record)
            return record

    def metrics(self, job_id: str) -> Optional[Dict[str, object]]:
        record = self.get(job_id)
        metrics_path = Path(record.output_dir) / "metrics.jsonl"
        if not metrics_path.exists():
            return None
        lines = [line for line in metrics_path.read_text().splitlines() if line]
        return json.loads(lines[-1]) if lines else None

    def log_tail(self, job_id: str, lines: int = 100) -> str:
        record = self.get(job_id)
        log_path = Path(record.log_path)
        if not log_path.exists():
            return ""
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])


def create_app(
    checkpoint: Path,
    *,
    device: Optional[str] = None,
    jobs_root: Path = Path("runs/detector_service"),
    api_key: Optional[str] = None,
):
    try:
        from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
        from fastapi.concurrency import run_in_threadpool
        from pydantic import BaseModel, Field
    except ImportError as error:
        raise RuntimeError("serving requires fastapi, uvicorn, and python-multipart") from error

    class TrainingRequest(BaseModel):
        dataset: str = Field(default="synthetic", pattern="^(synthetic|yolo)$")
        epochs: int = Field(default=20, ge=1, le=1000)
        batch_size: int = Field(default=16, ge=1, le=1024)
        image_size: int = Field(default=128, ge=32, le=2048)
        patch_size: int = Field(default=8, ge=2, le=64)
        seed: int = 42
        synthetic_samples: int = Field(default=4096, ge=1)
        validation_samples: int = Field(default=512, ge=1)
        validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
        vision_hidden_size: int = Field(default=128, ge=32)
        vision_layers: int = Field(default=4, ge=1)
        vision_heads: int = Field(default=4, ge=1)
        vision_expert_width: int = Field(default=48, ge=8)
        assignment_top_k: int = Field(default=5, ge=1, le=64)
        reg_max: int = Field(default=16, ge=0, le=32)
        head_hidden_size: int = Field(default=0, ge=0)
        dfl_loss_weight: float = Field(default=0.5, ge=0.0)
        quality_focal_beta: float = Field(default=2.0, ge=0.0)
        augmentation: str = Field(default="strong", pattern="^(light|strong)$")
        p2_head: bool = True
        stal: bool = True
        progressive_loss: bool = True
        momentum: float = Field(default=0.937, ge=0.0, lt=1.0)
        weight_decay: float = Field(default=5e-4, ge=0.0)
        learning_rate: float = Field(default=1e-2, gt=0.0)
        expert_lr_multiplier: float = Field(default=1.0, gt=0.0, le=10.0)
        yolo_images: Optional[str] = None
        yolo_labels: Optional[str] = None
        validation_yolo_images: Optional[str] = None
        validation_yolo_labels: Optional[str] = None
        backbone_checkpoint: Optional[str] = None
        detector_checkpoint: Optional[str] = None
        class_map: Optional[str] = None

    resolved_device = resolve_device(device)
    runtime = ModelRuntime(checkpoint, resolved_device)
    jobs = TrainingJobManager(jobs_root, sys.executable)
    app = FastAPI(title="TR-Hash Detector Service", version="0.4.0")

    def authenticate(x_api_key: Optional[str] = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {"status": "ok", **runtime.info()}

    @app.get("/v1/model", dependencies=[Depends(authenticate)])
    def model_info() -> Dict[str, object]:
        return runtime.info()

    @app.post("/v1/predict", dependencies=[Depends(authenticate)])
    async def predict(
        file: UploadFile = File(...),
        confidence: Optional[float] = None,
        iou_threshold: float = 0.45,
    ) -> Dict[str, object]:
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as error:
            raise HTTPException(status_code=400, detail="invalid image") from error
        started = time.perf_counter()
        selected_confidence = (
            float(confidence)
            if confidence is not None
            else float(runtime.info()["recommended_confidence"])
        )
        detections = await run_in_threadpool(
            runtime.predict,
            image,
            confidence=selected_confidence,
            iou_threshold=iou_threshold,
        )
        return {
            "detections": detections,
            "image": {"width": image.width, "height": image.height},
            "confidence": selected_confidence,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    @app.post("/v1/train", dependencies=[Depends(authenticate)])
    def start_training(request: TrainingRequest) -> Dict[str, object]:
        try:
            return asdict(jobs.start(request, str(resolved_device)))
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/train/{job_id}", dependencies=[Depends(authenticate)])
    def training_status(job_id: str) -> Dict[str, object]:
        try:
            return {**asdict(jobs.get(job_id)), "latest_metrics": jobs.metrics(job_id)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.get("/v1/train/{job_id}/logs", dependencies=[Depends(authenticate)])
    def training_logs(job_id: str, lines: int = 100) -> Dict[str, object]:
        try:
            return {"job_id": job_id, "log": jobs.log_tail(job_id, min(max(lines, 1), 1000))}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.delete("/v1/train/{job_id}", dependencies=[Depends(authenticate)])
    def cancel_training(job_id: str) -> Dict[str, object]:
        try:
            return asdict(jobs.cancel(job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/train/{job_id}/promote", dependencies=[Depends(authenticate)])
    def promote_checkpoint(job_id: str) -> Dict[str, object]:
        try:
            record = jobs.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        if record.status != "completed":
            raise HTTPException(status_code=409, detail="job is not completed")
        best = Path(record.output_dir) / "best"
        if not best.exists():
            raise HTTPException(status_code=409, detail="job has no validated checkpoint")
        runtime.load(best)
        return runtime.info()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jobs-root", type=Path, default=Path("runs/detector_service"))
    parser.add_argument("--api-key", default=os.environ.get("TR_HASH_API_KEY"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("serving requires uvicorn") from error
    uvicorn.run(
        create_app(
            args.checkpoint,
            device=args.device,
            jobs_root=args.jobs_root,
            api_key=args.api_key,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
