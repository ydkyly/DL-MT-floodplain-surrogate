import os
import sys

# --- 全局配置区域 ---

# 1. 【重要】指定包含所有子文件夹的“根目录”路径
#    脚本将处理这个根目录下的一级子文件夹。
#    - Windows 示例: r"D:\我的所有假期照片"
#
#    --- !!! 请在这里修改路径 !!! ---
ROOT_FOLDER_PATH = r"D:\Work\qyb\ResNet-18\data\train\rename"
#    --- !!! 请在这里修改路径 !!! ---


# 2. 文件名前缀 (如果不需要前缀，请留空)
FILE_PREFIX = ""

# 3. 每个子文件夹内，编号的起始数字
START_NUMBER = 30

# 4. 编号的位数 (例如，设置为 8 会生成 00000001, 00000002...)
ZERO_PADDING = 2

# 5. 是否只处理特定类型的文件 (留空列表[]表示处理所有文件)
#    例如: TARGET_EXTENSIONS = [".jpg", ".png"]
TARGET_EXTENSIONS = [".hdf"]


# --- 核心功能函数，通常无需修改 ---

def rename_files_in_single_folder(target_folder_path):
    """
    对单个指定的文件夹内的文件进行重命名。
    这个函数会被主程序为每个子文件夹调用一次。
    """
    print(f"\n--- 开始处理子文件夹: {target_folder_path} ---")

    try:
        # 获取文件夹中的所有项目（包括文件和可能存在的下级文件夹）
        all_items = os.listdir(target_folder_path)
        # [重要] 过滤出文件，忽略文件夹，避免对文件夹本身进行操作
        filenames = [f for f in all_items if os.path.isfile(os.path.join(target_folder_path, f))]
    except OSError as e:
        print(f"  [错误] 无法访问文件夹。原因: {e}")
        return  # 跳过这个有问题的子文件夹

    # [重要] 对文件名进行排序，以确保重命名顺序是可预测的 (按字母顺序)
    filenames.sort()

    # 如果指定了文件类型，则进行过滤
    if TARGET_EXTENSIONS:
        target_exts_lower = [ext.lower() for ext in TARGET_EXTENSIONS]
        original_count = len(filenames)
        filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in target_exts_lower]
        print(f"  检测到 {original_count} 个文件，根据扩展名过滤后，将处理 {len(filenames)} 个。")

    if not filenames:
        print("  该子文件夹中没有找到符合条件的文件，跳过。")
        return

    print(f"  准备重命名 {len(filenames)} 个文件...")

    # 计数器在每次函数调用时都会重置为 START_NUMBER
    current_number = START_NUMBER
    for old_filename in filenames:
        _, file_extension = os.path.splitext(old_filename)

        # 构建新的文件名
        new_filename = f"{FILE_PREFIX}{current_number:0{ZERO_PADDING}d}{file_extension}"

        old_filepath = os.path.join(target_folder_path, old_filename)
        new_filepath = os.path.join(target_folder_path, new_filename)

        if os.path.exists(new_filepath):
            print(f"  [警告] 跳过 '{old_filename}' 因为目标文件名 '{new_filename}' 已存在。")
            continue

        try:
            os.rename(old_filepath, new_filepath)
            print(f"  成功: '{old_filename}'  ->  '{new_filename}'")
            current_number += 1
        except OSError as e:
            print(f"  [错误] 重命名 '{old_filename}' 失败。原因: {e}")

    print(f"--- 子文件夹 {os.path.basename(target_folder_path)} 处理完成 ---")


def main():
    """
    主函数，负责遍历根目录下的所有子文件夹。
    """
    # 在开始前，检查根路径是否已经被修改
    if "YOUR_ROOT_FOLDER_PATH_HERE" in ROOT_FOLDER_PATH:
        print("错误: 你还没有在脚本中指定要处理的根目录路径。")
        print(r'请打开脚本文件，找到 ROOT_FOLDER_PATH = r"..." 这一行，并设置你的实际路径。')
        return

    print(f"*** 批量重命名任务开始 ***")
    print(f"指定的根目录是: {ROOT_FOLDER_PATH}")

    # 检查根目录是否存在
    if not os.path.isdir(ROOT_FOLDER_PATH):
        print(f"错误: 根目录 '{ROOT_FOLDER_PATH}' 不存在或路径不正确。")
        return

    # 获取根目录下的所有项目
    try:
        all_items_in_root = os.listdir(ROOT_FOLDER_PATH)
    except OSError as e:
        print(f"错误: 无法访问根目录。原因: {e}")
        return

    # 筛选出所有的子文件夹
    subfolders_to_process = [
        os.path.join(ROOT_FOLDER_PATH, item)
        for item in all_items_in_root
        if os.path.isdir(os.path.join(ROOT_FOLDER_PATH, item))
    ]

    if not subfolders_to_process:
        print("在指定的根目录下没有找到任何子文件夹。程序结束。")
        return

    print(f"发现 {len(subfolders_to_process)} 个子文件夹，将对它们逐一进行处理。")

    # 在执行前发出最后的警告
    print("\n!!! 重要警告 !!!")
    print("该脚本将永久性地修改所有找到的子文件夹内的文件名。")
    print("强烈建议您先在备份文件夹中进行测试。")

    choice = input("你确定要继续吗? (输入 'yes' 继续): ")

    if choice.lower() == 'yes':
        # 遍历每一个子文件夹，并调用重命名函数
        for folder_path in subfolders_to_process:
            rename_files_in_single_folder(folder_path)
        print("\n*** 所有任务处理完毕 ***")
    else:
        print("操作已取消。")


# --- 程序入口 ---
if __name__ == "__main__":
    main()