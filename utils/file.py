import os
import shutil
class FileUtil(object):
    def __init__(self,srcpath,tarpath):
        self.srcpath = srcpath
        self.tarpath = tarpath
        self.srcfile = ''
    def containfile(self):

        Files = os.listdir(self.srcpath)
        print("Files is  ",Files)
        for file in Files:
            if '.MP4' in file or '.mp4' in file:
                print(file)
                self.srcfile = file
                return file
        return ""


    def mvfile(self):
        print('开始移动文件')
        print('源文件所在目录： ',self.srcpath)
        print('目标文件所在目录： ',self.tarpath)

        src = os.path.join(self.srcpath, self.srcfile)
        dst = os.path.join(self.tarpath, self.srcfile)
        shutil.move(src, dst)
        print('移动完成')


if __name__ == '__main__':
    srcpath = 'C:\\Users\\Administrator\\Desktop\\flie\\src'
    srcfile = '简历_重邮_侯振东_P1.pdf'
    tarpath = 'C:\\Users\\Administrator\\Desktop\\flie\\tar'
    FileU = FileUtil(srcpath,tarpath)
    fname = FileU.containfile()
    print(fname)