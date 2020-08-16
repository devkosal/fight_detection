# This script takes raw videos in the specificed format below and converts them into frames
"""Format
data
--hockey
----fight
----nofght
--movies
----fight
----nofght
--surv
----fight
----nofght
"""

from devai.scripts.vid2frames import videos_to_frames_converter
from devai import Path

datasets = ["hockey", "movies", "surv"]


def main():
    data_path = Path("data/")

    for ds in datasets:
        for cl in ["fight", "nofight"]:
            current_path = data_path/ds/cl
            videos_to_frames_converter(current_path)


if __name__ == "__main__":
    main()
