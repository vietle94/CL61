from cl61.fetch.raw import fetch_raw, fetch_model
import os

for date in [
    "2023-07-02",
    "2023-07-03",
    "2023-09-13",
    "2023-09-14",
    "2023-11-10",
    "2023-11-11",
    "2024-01-24",
    "2024-01-25",
    "2024-03-25",
    "2024-03-26",
    "2024-05-08",
    "2024-05-09",
    "2024-07-13",
    "2024-07-14",
    "2024-09-15",
    "2024-09-16",
    "2024-11-27",
    "2024-11-28",
]:
    directory = "/media/viet/CL61/studycase/kenttarova/" + date.replace("-", "")

    # Check if directory exists, otherwise create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
        os.makedirs(directory + "/weather/")
    else:
        print(f"Directory already exists: {directory}")
        continue
    fetch_raw(
        "kenttarova",
        date,
        date,
        directory + "/",
    )
    fetch_model(
        "kenttarova",
        date,
        date,
        directory + "/weather/",
    )
