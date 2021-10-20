using Microsoft.Win32;

using OpenCvSharp;

using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

using static OpenCvSharp.Cv2;
using static OpenCvSharp.Mat;
using static OpenCvSharp.MatType;
using static System.Math;


namespace EdgeDetection_Sample {
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window {

        #region Ctr

        public MainWindow() {
            InitializeComponent();

            this._inputMat = new Mat();
            this._outputMat = new Mat();
            this.MatConvertFunc = () => this.InputMat.Clone();
        }

        #endregion Ctr

        #region type

        private static readonly MatType DoubleOne = CV_64FC1;
        private static readonly MatType ByteOne = CV_8UC1;

        #endregion

        #region Prop

        private Mat _inputMat;
        private Mat _outputMat;

        private Func<Mat> MatConvertFunc { get; set; }

        private Mat InputMat {
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

        private Mat OutputMat {
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

        #endregion Prop

        #region Event

        private void Open_MenuItem_OnClick(Object sender, RoutedEventArgs e) {
            OpenFileDialog dialog = new();
            if (dialog.ShowDialog(this) is false) {
                return;
            }

            var fileStream = dialog.OpenFile();
            this.InputMat = FromStream(fileStream, ImreadModes.Color);
        }

        private void Convert_Button_OnClick(Object sender, RoutedEventArgs e) {
            this.OutputMat = this.MatConvertFunc.Invoke();
        }

        private unsafe Mat Marr(Mat src, Int32 kernelDiameter, Double sigma) {
            var size = kernelDiameter / 2;
            Mat kernel = new(kernelDiameter, kernelDiameter, DoubleOne);

            var pSrc = (Byte*)src.Ptr().ToPointer();
            var pKernel = (Double*)kernel.Ptr().ToPointer();

            for (var i = -size; i <= size; i++) {
                for (var j = -size; j <= size; j++) {
                    pKernel[j + size + 2 * size * (i + size)] =
                        Exp(-((Pow(j, 2) + Pow(i, 2)) / (Pow(sigma, 2) * 2))) * ((Pow(j, 2) + Pow(i, 2) - 2 * Pow(sigma, 2)) / (2 * Pow(sigma, 4)));
                }
            }

#if SLOW_MODE

                Mat laplacianMat = new(src.Rows - size * 2, src.Cols - size * 2, DoubleOne);
                Mat dst = Zeros(src.Rows - size * 2, src.Cols - size * 2, ByteOne);
                var pLaplacian = (Double*)laplacianMat.Ptr().ToPointer();
                for (var i = size; i < src.Rows - size; i++)
                {
                    for (var j = size; j < src.Cols - size; j++)
                    {
                        Double sum = 0;
                        for (var x = -size; x <= size; x++)
                        {
                            for (var y = -size; y <= size; y++)
                            {
                                sum += pKernel[y + size + (x + size) * 2 * size] * pSrc[j + y + (i + x) * src.Cols];
                            }
                        }
                        pLaplacian[j - size + (i - size) * (src.Cols - 2 * size)] = sum;
                    }
                }

#else // SLOW_MODE

            Mat laplacianMat = Zeros(src.Size(), DoubleOne);
            Mat dst = Zeros(laplacianMat.Size(), ByteOne);
            Filter2D(src, laplacianMat, DoubleOne, kernel);
            var pLaplacian = (Double*)laplacianMat.Ptr().ToPointer();

#endif // SLOW_MODE

            var pDst = (Byte*)dst.Ptr().ToPointer();
            for (var i = 1; i < dst.Rows - 1; i++) {
                for (var j = 1; j < dst.Cols - 1; j++) {
                    if (pLaplacian[j + (i - 1) * dst.Cols] * pLaplacian[j + (i + 1) * dst.Cols] < 0 ||
                        pLaplacian[j - 1 + (i - 1) * dst.Cols] * pLaplacian[j + 1 + (i + 1) * dst.Cols] < 0 ||
                        pLaplacian[j + 1 + (i - 1) * dst.Cols] * pLaplacian[j - 1 + (i + 1) * dst.Cols] < 0 ||
                        pLaplacian[j + 1 + i * dst.Cols] * pLaplacian[j - 1 + i * dst.Cols] < 0) {
                        pDst[j + i * dst.Cols] = Byte.MaxValue;
                    }
                }
            }

            return dst;
        }

        private Mat ProcessMat(Action<Mat> processAction) {
            Mat[] srcs = this.InputMat.CvtColor(ColorConversionCodes.BGR2HLS).Split();
            using Mat dst = new();
            processAction.Invoke(srcs[1]);
            Merge(srcs, dst);
            Mat resultMat = dst.CvtColor(ColorConversionCodes.HLS2BGR);
            foreach (var src in srcs) {
                src.Dispose();
            }
            return resultMat;
        }

        private void EdgeDetection_OnClick(Object sender, RoutedEventArgs e) {
            var senderMenuItem = sender as MenuItem;
            this.activeFunction_TextBlock.Text = (senderMenuItem?.Header as String)?[1..^0];
            this.MatConvertFunc = (senderMenuItem?.Tag as String) switch {
                "MG" => () => this.InputMat.MorphologyEx(MorphTypes.Gradient, null),
                "S" => () => ProcessMat(src => Sobel(src, src, src.Type(), 1, 1)),
                "L" => () => ProcessMat(src => Laplacian(src, src, src.Type())),
                "M" => () => {
                    Mat[] srcs = this.InputMat.CvtColor(ColorConversionCodes.BGR2HLS).Split();
                    var dst = Marr(srcs[1], 9, 1.6);
                    foreach (var src in srcs) {
                        src.Dispose();
                    }
                    Mat resultMat = dst.CvtColor(ColorConversionCodes.GRAY2BGR);

                    return resultMat;
                }
                ,
                "C" => () => ProcessMat(src => Canny(src, src, 150, 100)),
                _ => throw new ArgumentOutOfRangeException(nameof(sender), "")
            };
        }

        #endregion Event
    }
}