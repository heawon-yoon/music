import os



def get_full_path(file_path):
    file_dir = os.path.dirname(file_path)
    file_name_with_extension = os.path.basename(file_path)
    file_name = os.path.splitext(file_name_with_extension)[0]

    full_path = os.path.join(file_dir, file_name)

    return full_path