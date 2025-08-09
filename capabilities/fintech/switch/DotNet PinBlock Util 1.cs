using Microsoft.Extensions.Logging;
using Org.BouncyCastle.Crypto.Engines;
using Org.BouncyCastle.Crypto.Parameters;
using System.Diagnostics;
using System.Security.Cryptography;
using Zone.Card.Core.Communications;
using Zone.Iso8583.OpenIso8583;

namespace Zone.Card.Utilities
{
    internal class Program
    {
        private static int _mode;

        static async Task Main(string[] args)
        {
            Console.WriteLine("STARTING ZONE CARD UTILITIES CONSOLE");
            Console.WriteLine("CHOOSE YOUR FIGHTER");
            Console.WriteLine("2) GENERATE CLEAR ZPK.    3) GENERATE ENCRYPTED PIN BLOCK.");
            string mode = Console.ReadLine();
            _mode = int.Parse(mode);
            
			if (_mode == 2)
                ClearZpkGenerationMode();
            else if (_mode == 3)
                EncryptedPinBlockGenerationMode();
        }



        static void ClearZpkGenerationMode()
        {
            while (true)
            {
                try
                {
                    Console.WriteLine("Input ZMK");
                    string zmk = Console.ReadLine();

                    Console.WriteLine("Input Encrypted ZPK");
                    string zpk = Console.ReadLine();

                    string encryptedZpkPartA = zpk[..16];
                    string encryptedZpkPartB = zpk.Substring(16, 16);

                    byte[] encryptedZmkPartB = StringToByteArray(zmk.Substring(16, 16));

                    string zmkPartBVariant1 = ByteArrayToString(XorIt(encryptedZmkPartB, StringToByteArray("A6".PadRight(16, '0'))));
                    string zmkVariant1 = zmk[..16] + zmkPartBVariant1;

                    string zmkPartBVariant2 = ByteArrayToString(XorIt(encryptedZmkPartB, StringToByteArray("5A".PadRight(16, '0'))));
                    string zmkVariant2 = zmk[..16] + zmkPartBVariant2;

                    string result1 = Operate(StringToByteArray(encryptedZpkPartA), StringToByteArray(zmkVariant1), false);

                    string result2 = Operate(StringToByteArray(encryptedZpkPartB), StringToByteArray(zmkVariant2), false);

                    string clearZpk = result1 + result2;
                    Console.WriteLine($"Clear ZPK is {clearZpk}");

                    string kcv = new string(Operate(new byte[8], StringToByteArray(clearZpk), true).Take(..6).ToArray());
                    Console.WriteLine($"Clear ZPK KCV {kcv}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Silencing exception {ex}, that occured during Clear ZPK Generation");
                }
            }
        }

        static void EncryptedPinBlockGenerationMode()
        {
            while (true)
            {
                try
                {
                    Console.WriteLine("Input Clear ZPK");
                    string zpk = Console.ReadLine();

                    Console.WriteLine("Input Card PAN");
                    string cardPan = Console.ReadLine();

                    Console.WriteLine("Input PIN");
                    string pin = Console.ReadLine();

                    var pinString = $"0{pin.Length}{pin}".PadRight(16, 'F');

                    var pinBlock1 = StringToByteArray(pinString);

                    var treatedPan = cardPan.Substring(cardPan.Length - 13, 12).PadLeft(16, '0');
                    Console.WriteLine($"Account Number: {cardPan.Substring(cardPan.Length - 13, 12)}");
                    var pinBlock2 = StringToByteArray(treatedPan);

                    var clearPinBlock = XorIt(pinBlock1, pinBlock2);
                    Console.WriteLine($"Clear Pin Block: {ByteArrayToString(clearPinBlock)}");

                    var encryptedPinBlock = Operate(clearPinBlock, StringToByteArray(zpk), true);
                    Console.WriteLine($"Encrypted Pin Block: {encryptedPinBlock}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Silencing exception {ex}, that occured during Encrypted Clear PIN Block Generation");
                }
            }
        }

        public static string ByteArrayToString(byte[] bytes)
        {
            return BitConverter.ToString(bytes).Replace("-", "");
        }

        public static byte[] StringToByteArray(string str)
        {
            int numberChars = str.Length;

            byte[] bytes = new byte[numberChars / 2];

            for (int i = 0; i < numberChars; i += 2)
                bytes[i / 2] = Convert.ToByte(str.Substring(i, 2), 16);


            return bytes;
        }

        public static byte[] XorIt(byte[] input1, byte[] input2)
        {
            byte[] bytes = new byte[input2.Length];

            for (int i = 0; i < input2.Length; i++) bytes[i] = (byte)(input2[i] ^ input1[i % input1.Length]);


            return bytes;
        }

        static string Operate(byte[] data, byte[] key, bool isEncrypt)
        {
            TripleDES tdes = TripleDES.Create();
            tdes.Key = key;
            tdes.Mode = CipherMode.ECB;
            tdes.Padding = PaddingMode.None;

            ICryptoTransform cryptoTransform;
            if (isEncrypt)
                cryptoTransform = tdes.CreateEncryptor();
            else
                cryptoTransform = tdes.CreateDecryptor();

            byte[] resultArray = cryptoTransform.TransformFinalBlock(data, 0, data.Length);            
            tdes.Clear();

            return ByteArrayToString(resultArray);
        }       
    }
}
