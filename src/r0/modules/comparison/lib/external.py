"""Explicit external worker/configuration; execution files are bound automatically."""

from dataclasses import dataclass
from pathlib import Path

from .storage import Document, file_hash, read_json


@dataclass(frozen=True)
class ExternalPredictor:
    name: str
    python: Path
    config: Path
    worker: Path | None = None  # None selects the built-in TabPFN worker.
    artifacts: tuple[Path, ...] = ()

    def snapshot(self) -> Document:
        if self.name not in ("learner", "tabpfn") or (
            self.worker is None and self.name != "tabpfn"
        ):
            raise ValueError("external role must be learner or tabpfn with an explicit worker")
        identity = read_json(self.config)["identity"]
        if set(identity) != {"implementation", "environment", "settings", "checkpoint", "evidence"}:
            raise ValueError("incomplete adapter identity")
        if not all(identity.values()) or identity["evidence"] != "FIXTURE_ONLY":
            raise ValueError("real external execution requires the guarded admitted lane")
        paths = [self.python, self.config, *self.artifacts]
        if self.worker is not None:
            paths.append(self.worker)
        if any(not path.is_file() for path in paths):
            raise ValueError("provisioned interpreter, worker, config and artifacts must be files")
        return {
            "name": self.name,
            "identity": identity,
            "command": self.command(),
            "files": {str(path.absolute()): file_hash(path) for path in paths},
        }

    def command(self) -> list[str]:
        worker = (
            [str(self.worker.absolute())]
            if self.worker
            else [
                "-m",
                "r0.modules.comparison",
                "tabpfn-worker",
            ]
        )
        return [str(self.python.absolute()), *worker, "--config", str(self.config.absolute())]

    @classmethod
    def from_file(cls, path: Path) -> "ExternalPredictor":
        data = read_json(path)
        if set(data) - {"name", "python", "config", "worker", "artifacts"}:
            raise ValueError("external configuration uses name/python/config/worker/artifacts")

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else path.resolve().parent / candidate

        return cls(
            data["name"],
            resolve(data["python"]),
            resolve(data["config"]),
            resolve(data["worker"]) if data.get("worker") else None,
            tuple(resolve(p) for p in data.get("artifacts", [])),
        )
