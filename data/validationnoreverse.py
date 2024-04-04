import json

import requests
import os

folder_file_mapping = dict(
    # testing='final_testing_raw_data.json',
#    training='final_training_raw_data.json',
     validation='final_validation_raw_data.json'
)


def create_folders():
    print('Creating folders...')
    for name in folder_file_mapping.keys():
        try:
            os.makedirs(name)
        except FileExistsError:
            # directory already exists
            pass


def download_images():
    count, skipped = 0, 0
    for _file in folder_file_mapping.items():
        folder_name, file_name = _file
        print(f"Selected folder = {folder_name}")
        print(f"Selected file = {file_name}")
        with open(f"{file_name}", 'r') as current_file:
            json_data = json.load(current_file)
            ind_count, ind_skipped = 0, 0
            for _data in json_data:
                name = _data.get('name')
                link = _data.get('url')
                r = requests.get(link).content
                _img_name = name.split('/')[-1]
                if os.path.exists(f"{folder_name}/{_img_name}"):
                    ind_skipped += 1
                    ind_count += 1
                    print(f'Duplicate file {_img_name}.\nSkipping download. Skip count: {ind_skipped}\nDownload Count: {ind_count}/{len(json_data) + 1}')
                    continue
                with open(f"{folder_name}/{_img_name}", "wb+") as f:
                    ind_count += 1
                    print(f'Downloading and saving {_img_name} in {folder_name}.\nDownload Count: {ind_count}/{len(json_data) + 1}')
                    f.write(r)
        count += ind_count
        skipped += ind_skipped

    print(f"Total Downloaded: {count}, Total Skipped: {skipped}")


# Press the green button in the gutter to run the script.
# run the virtual environment: source myenv/bin/activate
if __name__ == '__main__':
    create_folders()
    download_images()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
