# 看下文件夹层级结构
import os
import os.path


def dfs_showdir(path, depth):
    if depth == 0:
        print("root:[" + path + "]")
    # print("当前文件路径是{}，包含文件有{}。".format(path, os.listdir(path)))

    for item in os.listdir(path):
        if item in ['.git', '.idea', '__pycache__']:
            continue

        print("| " * depth + "+--" + item)

        new_item = path + '/' + item
        if os.path.isdir(new_item):
            dfs_showdir(new_item, depth + 1)


if __name__ == '__main__':
    dfs_showdir('..', 0)
