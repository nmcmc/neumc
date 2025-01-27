import tempfile
from shutil import move

import torch


def save_checkpoint(*, model, optimizer, scheduler=None, era, path, **kwargs):
    to_save = {
        "era": era,
        "state_dict": model.state_dict(),
        "optimizer": type(optimizer).__name__,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    to_save |= kwargs
    if scheduler:
        to_save |= {
            "scheduler": type(scheduler).__name__,
            "scheduler_state_dict": scheduler.state_dict(),
        }
    else:
        to_save |= {
            "scheduler": "",
            "scheduler_state_dict": "",
        }

    torch.save(
        to_save,
        path,
    )


def safe_save_checkpoint(
    *, model, optimizer, scheduler, era, path, tmp_dir=None, **kwargs
):
    tmpfile = tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        era=era,
        path=tmpfile,
        **kwargs,
    )
    tmpfile.close()
    move(tmpfile.name, path)
