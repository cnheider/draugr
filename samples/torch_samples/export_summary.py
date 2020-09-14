from pathlib import Path

from apppath import ensure_existence
from matplotlib import pyplot

from draugr import PROJECT_APP_PATH
from draugr.tensorboard_utilities import TensorboardEventExporter
from draugr.writers.standard_tags import TrainingScalars

if __name__ == "__main__":
    save = False
    event_files = list(PROJECT_APP_PATH.user_log.rglob("events.out.tfevents.*"))
    if len(event_files) > 0:
        for _path_to_events_file in event_files:
            print(f"Event file: {_path_to_events_file}")
            _out_dir = Path.cwd() / "exclude" / "results"
            ensure_existence(_out_dir)
            tee = TensorboardEventExporter(
                _path_to_events_file.parent, save_to_disk=save
            )
            print(f"Available tags: {tee.tags_available}")
            tee.export_line_plot(TrainingScalars.training_loss.value, out_dir=_out_dir)
            if not save:
                pyplot.show()
    else:
        print("No events found")
