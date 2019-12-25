import os


def generate():
    with open('/Users/yangchao/data/SIXray/Testset/test.txt', 'w') as f:
        content = ''
        for root, dirs, files, in os.walk('/Users/yangchao/data/SIXray/Annotation/'):
            i, j = 1, 1
            for file in files:
                if i <= 100:
                    if os.path.splitext(file)[0].startswith('core_battery') and os.path.splitext(file)[1] == '.txt':
                        content += os.path.splitext(file)[0] + '\n'
                        i += 1
                if j <= 100:
                    if os.path.splitext(file)[0].startswith('coreless_battery') and os.path.splitext(file)[1] == '.txt':
                        content += os.path.splitext(file)[0] + '\n'
                        j += 1
            f.write(content)


def generate1(targetpath, imgpath):
    with open(targetpath + '/test.txt', 'w') as f:
        content = ''
        for root, dirs, files, in os.walk(imgpath):
            for file in files:
                content += os.path.splitext(file)[0] + '\n'
            f.write(content)


if __name__ == '__main__':
    # generate1('/Users/yangchao/data/SIXray/Testset', '/Users/yangchao/data/SIXray/Image')
    generate()
    # i = 1
    # for item in count11():
    #     print(i, '=', item)
    #     i += 1
