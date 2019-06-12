# ThesisTask
Программа для построения трехмерных моделей по коллекциям изображений

# Инструкция по компиляции 
Дальнейшие указания приведены для системы Ubuntu.

## Подключение OpenCV

Необходимо выполнить следующую последовательность консольных команд:

### 0. Перейти в произвольную папку
```
cd ~
```

### 1. Выбрать версию OpenCV для установки
```
cvVersion="3.4.4"
```
Создаем папку, в которую будет производиться установка библиотеки.

```
mkdir installation
mkdir installation/OpenCV-"$cvVersion"
```
Сохраняем текущую рабочую папку в переменной cwd (OpenCV_Home_Dir)
```
cwd=$(pwd)
```
### 2. Обновить пакеты

```
sudo apt -y update
sudo apt -y upgrade
```

### 3. Установить системные библиотеки

```
sudo apt -y remove x264 libx264-dev

sudo apt -y install build-essential checkinstall cmake pkg-config yasm
sudo apt -y install git gfortran
sudo apt -y install libjpeg8-dev libjasper-dev libpng12-dev
 
sudo apt -y install libtiff5-dev
 
sudo apt -y install libtiff-dev
 
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt -y install libxine2-dev libv4l-dev
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd $cwd
 
sudo apt -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt -y install libgtk2.0-dev libtbb-dev qt5-default
sudo apt -y install libatlas-base-dev
sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt -y install libvorbis-dev libxvidcore-dev
sudo apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt -y install libavresample-dev
sudo apt -y install x264 v4l-utils

sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install libgoogle-glog-dev libgflags-dev
sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

sudo apt-get install libgoogle-glog-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libeigen3-dev
```

### 4. Скачать opencv и opencv_contrib

```
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout $cvVersion
cd ..
 
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout $cvVersion
cd ..
```

### 5. Скомпилировать OpenCV с модулями из opencv_contrib

Создаем и переходим в папку build

```
cd opencv
mkdir build
cd build
```

Компиляция OpenCV
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion" \
            -D INSTALL_C_EXAMPLES=ON \
            -D WITH_TBB=ON \
            -D WITH_V4L=ON \
        -D WITH_QT=ON \
        -D WITH_OPENGL=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON ..
        
make -j4
make install
```

## Компиляция самой программы 

### Установка cmake

```
sudo add-apt-repository ppa:george-edison55/cmake-3.x
sudo apt-get update

sudo apt-get install cmake
```
### Основная часть

Необходимо скачать main.cpp и CMakeLists.txt из этого репозитория, поместить их в выбранную папку.

Далее, создадим папку build и перейдем в нее

```
mkdir build

cd build
```
Компиляция
```
cmake ..

make
```

В папке build появится готовый к использованию исполняемый файл ThesisTask
