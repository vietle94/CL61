from cl61.fetch.raw import fetch_raw, fetch_model
import os

for date in [
    "2022-06-24",
    "2022-06-25",
    "2022-08-14",
    "2022-08-15",
    "2023-02-10",
    "2023-02-11",
    "2023-03-16",
    "2023-03-17",
    "2023-04-15",
    "2023-04-16",
    "2023-05-23",
    "2023-05-24",
    "2023-06-15",
    "2023-06-16",
    "2023-08-07",
    "2023-08-08",
    "2023-10-20",
    "2023-10-21",
    "2024-03-18",
    "2024-03-19",
    "2024-05-08",
    "2024-05-09",
    "2024-06-22",
    "2024-06-23",
    "2024-07-21",
    "2024-07-22",
    "2024-09-04",
    "2024-09-05",
    "2024-11-09",
    "2024-11-10",
]:
    directory = "/media/viet/CL61/studycase/vehmasmaki/" + date.replace("-", "")

    # Check if directory exists, otherwise create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
        os.makedirs(directory + "/weather/")
    else:
        print(f"Directory already exists: {directory}")
        continue
    fetch_raw(
        "vehmasmaki",
        date,
        date,
        directory + "/",
    )
    fetch_model(
        "vehmasmaki",
        date,
        date,
        directory + "/weather/",
    )
