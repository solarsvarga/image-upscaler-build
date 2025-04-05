# AI Image Upscaler для Windows 7
# Полная версия с поддержкой CPU/CUDA
import sys
import os
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QFileDialog, 
                            QCheckBox, QGroupBox, QProgressBar, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class UpscaleWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path, image_path, scale, denoise_strength, face_enhance):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.scale = scale
        self.denoise_strength = denoise_strength
        self.face_enhance = face_enhance
        self._is_running = True

    def run(self):
        try:
            # Загрузка изображения
            self.progress_updated.emit(10, "Загрузка изображения...")
            img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Не удалось загрузить изображение")

            # Конвертация цветового пространства
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Инициализация модели
            self.progress_updated.emit(20, "Загрузка модели ИИ...")
            model = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23, 
                num_grow_ch=32, 
                scale=self.scale
            )
            
            upsampler = RealESRGANer(
                scale=self.scale,
                model_path=self.model_path,
                model=model,
                dni_weight=self.denoise_strength,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=False,  # Отключено для стабильности на Win7
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Улучшение лиц (если включено)
            if self.face_enhance:
                from gfpgan import GFPGANer
                face_enhancer = GFPGANer(
                    model_path='models/GFPGANv1.3.pth',
                    upscale=self.scale,
                    arch='clean',
                    channel_multiplier=2,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

            # Обработка изображения
            self.progress_updated.emit(40, "Обработка изображения...")
            output, _ = upsampler.enhance(img, outscale=self.scale)

            # Улучшение лиц
            if self.face_enhance:
                self.progress_updated.emit(60, "Улучшение лиц...")
                _, _, output = face_enhancer.enhance(
                    output,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )

            self.progress_updated.emit(90, "Финализация...")
            self.result_ready.emit(output)
            self.progress_updated.emit(100, "Готово!")

        except Exception as e:
            self.error_occurred.emit(f"{str(e)}\n\nДетали:\n{traceback.format_exc()}")
        finally:
            self._is_running = False

    def stop(self):
        self._is_running = False
        self.quit()

class ImageUpscalerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Upscaler для Windows 7")
        self.setWindowIcon(QIcon('icon.ico'))
        self.setGeometry(100, 100, 900, 600)
        
        # Модели
        self.model_paths = {
            "RealESRGAN (универсальная)": "models/RealESRGAN_x4plus.pth",
            "RealESRGAN (аниме)": "models/RealESRGAN_x4plus_anime_6B.pth",
            "ESRGAN": "models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"
        }
        
        self.current_image = None
        self.upscaled_image = None
        self.worker = None
        
        self.init_ui()
        self.check_hardware()
        
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель - настройки
        settings_panel = QGroupBox("Настройки обработки")
        settings_layout = QVBoxLayout()
        
        # Информация о системе
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #666; font-size: 10pt;")
        settings_layout.addWidget(self.info_label)
        
        # Выбор модели
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_paths.keys())
        settings_layout.addWidget(QLabel("Модель ИИ:"))
        settings_layout.addWidget(self.model_combo)
        
        # Масштаб
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "3x", "4x"])
        settings_layout.addWidget(QLabel("Масштаб увеличения:"))
        settings_layout.addWidget(self.scale_combo)
        
        # Параметры обработки
        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setValue(50)
        settings_layout.addWidget(QLabel("Подавление шумов:"))
        settings_layout.addWidget(self.denoise_slider)
        
        self.face_enhance_check = QCheckBox("Улучшение лиц (для портретов)")
        settings_layout.addWidget(self.face_enhance_check)
        
        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Загрузить")
        self.load_btn.setIcon(QIcon('icons/folder.png'))
        self.load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_btn)
        
        self.upscale_btn = QPushButton("Обработать")
        self.upscale_btn.setIcon(QIcon('icons/process.png'))
        self.upscale_btn.clicked.connect(self.start_upscaling)
        self.upscale_btn.setEnabled(False)
        btn_layout.addWidget(self.upscale_btn)
        
        self.save_btn = QPushButton("Сохранить")
        self.save_btn.setIcon(QIcon('icons/save.png'))
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        settings_layout.addLayout(btn_layout)
        
        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.status_label = QLabel("Готов к работе")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        settings_layout.addWidget(self.progress_bar)
        settings_layout.addWidget(self.status_label)
        settings_panel.setLayout(settings_layout)
        main_layout.addWidget(settings_panel, stretch=1)
        
        # Правая панель - изображения
        image_panel = QWidget()
        image_layout = QVBoxLayout()
        
        # Оригинал
        orig_group = QGroupBox("Оригинал")
        orig_layout = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("background-color: #eee; border: 1px solid #ddd;")
        orig_layout.addWidget(self.original_label)
        orig_group.setLayout(orig_layout)
        image_layout.addWidget(orig_group)
        
        # Результат
        result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout()
        self.upscaled_label = QLabel()
        self.upscaled_label.setAlignment(Qt.AlignCenter)
        self.upscaled_label.setMinimumSize(400, 300)
        self.upscaled_label.setStyleSheet("background-color: #eee; border: 1px solid #ddd;")
        result_layout.addWidget(self.upscaled_label)
        result_group.setLayout(result_layout)
        image_layout.addWidget(result_group)
        
        image_panel.setLayout(image_layout)
        main_layout.addWidget(image_panel, stretch=2)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Таймер для проверки памяти
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_info)
        self.memory_timer.start(2000)
    
    def check_hardware(self):
        # Проверка оборудования
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        vram = ""
        
        if device == "CUDA":
            vram = f", VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
            
        self.info_label.setText(
            f"Режим: {device}{vram}\n"
            f"PyTorch: {torch.__version__}\n"
            f"Разрешение: {self.screen().size().width()}x{self.screen().size().height()}"
        )
    
    def update_memory_info(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()/1024**2
            reserved = torch.cuda.memory_reserved()/1024**2
            self.status_label.setText(f"VRAM: {allocated:.1f}/{reserved:.1f} MB")
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tif);;Все файлы (*.*)"
        )
        
        if file_path:
            self.current_image = file_path
            try:
                # Загрузка с поддержкой Unicode путей
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)  # Коррекция ориентации
                
                # Конвертация в QPixmap
                img = img.convert("RGB")
                data = img.tobytes("raw", "RGB")
                q_img = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Масштабирование с сохранением пропорций
                self.original_label.setPixmap(
                    pixmap.scaled(
                        self.original_label.width()-20,
                        self.original_label.height()-20,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
                
                self.upscale_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.progress_bar.setValue(0)
                self.status_label.setText("Изображение загружено")
                
                # Очистка предыдущего результата
                self.upscaled_label.clear()
                self.upscaled_image = None
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")
    
    def start_upscaling(self):
        if not self.current_image:
            return
            
        model_name = self.model_combo.currentText()
        model_path = self.model_paths.get(model_name)
        
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Ошибка", f"Модель {model_name} не найдена!\nПроверьте папку models")
            return
            
        scale = int(self.scale_combo.currentText()[:-1])
        denoise_strength = self.denoise_slider.value() / 100.0
        face_enhance = self.face_enhance_check.isChecked()
        
        # Блокировка интерфейса
        self.set_ui_enabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Начало обработки...")
        
        # Запуск обработки в отдельном потоке
        self.worker = UpscaleWorker(
            model_path,
            self.current_image,
            scale,
            denoise_strength,
            face_enhance
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.handle_upscale_result)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
    
    def set_ui_enabled(self, enabled):
        self.load_btn.setEnabled(enabled)
        self.upscale_btn.setEnabled(enabled and self.current_image is not None)
        self.save_btn.setEnabled(enabled and self.upscaled_image is not None)
        self.model_combo.setEnabled(enabled)
        self.scale_combo.setEnabled(enabled)
        self.denoise_slider.setEnabled(enabled)
        self.face_enhance_check.setEnabled(enabled)
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def handle_upscale_result(self, result):
        self.upscaled_image = result
        
        # Конвертация numpy array в QPixmap
        height, width = result.shape[:2]
        if result.ndim == 3:
            bytes_per_line = 3 * width
            q_img = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            q_img = QImage(result.data, width, height, QImage.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(q_img)
        self.upscaled_label.setPixmap(
            pixmap.scaled(
                self.upscaled_label.width()-20,
                self.upscaled_label.height()-20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        
        self.save_btn.setEnabled(True)
        self.status_label.setText("Обработка завершена!")
    
    def handle_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        self.status_label.setText("Ошибка при обработке")
        self.progress_bar.setValue(0)
    
    def on_worker_finished(self):
        self.set_ui_enabled(True)
        if self.progress_bar.value() == 100:
            self.status_label.setText("Готово!")
        else:
            self.status_label.setText("Прервано")
    
    def save_result(self):
        if self.upscaled_image is None:
            return
            
        default_path = os.path.splitext(self.current_image)[0] + "_upscaled.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                # Определение формата по расширению
                ext = os.path.splitext(file_path)[1].lower()
                
                # Конвертация BGR в RGB если нужно
                if len(self.upscaled_image.shape) == 3 and self.upscaled_image.shape[2] == 3:
                    save_img = cv2.cvtColor(self.upscaled_image, cv2.COLOR_RGB2BGR)
                else:
                    save_img = self.upscaled_image
                
                # Сохранение с учетом формата
                if ext in ('.jpg', '.jpeg'):
                    cv2.imwrite(file_path, save_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                elif ext == '.webp':
                    cv2.imwrite(file_path, save_img, [int(cv2.IMWRITE_WEBP_QUALITY), 90])
                else:
                    cv2.imwrite(file_path, save_img)
                
                QMessageBox.information(self, "Сохранено", f"Изображение успешно сохранено как {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def closeEvent(self, event):
        # Корректное завершение worker при закрытии
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        event.accept()

if __name__ == "__main__":
    # Настройка для Windows 7
    if sys.platform == 'win32':
        import ctypes
        # Включение масштабирования для высоких DPI
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Унифицированный стиль
    
    # Проверка версий
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    window = ImageUpscalerApp()
    window.show()
    sys.exit(app.exec_())