using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

using Microsoft.Win32;

using OpenCvSharp;

using static OpenCvSharp.Mat;

namespace EdgeDetection_Sample {

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window {

        public MainWindow() {
            InitializeComponent();

            this._inputMat = new Mat();
            this._outputMat = new Mat();
            this.matConvertFunc = () => this.inputMat.Clone();
        }

        private Mat _inputMat;
        private Mat _outputMat;

        private Func<Mat> matConvertFunc { get; set; }

        private Mat inputMat {
            get => this._inputMat;
            set {
                if (ReferenceEquals(this._inputMat, value)) {
                    return;
                }
                this._inputMat.Dispose();
                this._inputMat = value;
                UpdateImage(this._inputMat, this.input_Image);
            }
        }

        private Mat outputMat {
            get => this._outputMat;
            set {
                if (ReferenceEquals(this._outputMat, value)) {
                    return;
                }
                this._outputMat.Dispose();
                this._outputMat = value;
                UpdateImage(this._outputMat, this.output_Image);
            }
        }

        public static void UpdateImage(Mat inputMat, Image image) {
            if (inputMat.Empty()) {
                return;
            }
            BitmapImage bitmapImage = new();
            bitmapImage.BeginInit();
            {
                bitmapImage.StreamSource = inputMat.ToMemoryStream();
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            }
            bitmapImage.EndInit();
            image.Source = bitmapImage;
        }

        #region Event

        private void Open_MenuItem_OnClick(Object sender, RoutedEventArgs e) {
            OpenFileDialog dialog = new();
            if (dialog.ShowDialog(this) is false) {
                return;
            }

            var fileStream = dialog.OpenFile();
            this.inputMat = FromStream(fileStream, ImreadModes.Color);
        }

        #endregion Event

        private void Convert_Button_OnClick(Object sender, RoutedEventArgs e) {
            this.outputMat = this.matConvertFunc.Invoke();
        }
    }
}