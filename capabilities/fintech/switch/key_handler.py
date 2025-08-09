from Crypto.Cipher import DES3
import binascii


def triple_des_decrypt(data, key):
    """Decrypt data using 3DES in ECB mode."""
    if len(key) != 32:  # 16 bytes in hex = 32 characters
        raise ValueError(f"Invalid key length: {len(key)}, expected 32")
    if len(data) != 16:  # 8 bytes in hex = 16 characters
        raise ValueError(f"Invalid data length: {len(data)}, expected 16")

    # Convert hex strings to bytes
    key_bytes = binascii.unhexlify(key)
    data_bytes = binascii.unhexlify(data)

    # Create cipher object
    cipher = DES3.new(key_bytes, DES3.MODE_ECB)

    # Decrypt and return full result (32 chars)
    decrypted = cipher.decrypt(data_bytes)
    return binascii.hexlify(decrypted).decode() + "0" * 16


def compute_clear_zpk(encrypted_zpk, encrypted_zmk):
    """
    Compute clear ZPK using encrypted ZPK and ZMK.
    """
    print("\nComputing Clear ZPK:")
    print(f"Encrypted ZPK: {encrypted_zpk}")
    print(f"Encrypted ZMK: {encrypted_zmk}")

    # Split parts
    zpk_part_a = encrypted_zpk[:16]
    zpk_part_b = encrypted_zpk[16:]
    zmk_part_a = encrypted_zmk[:16]
    zmk_part_b = encrypted_zmk[16:]

    print(f"\nZPK parts: A={zpk_part_a} B={zpk_part_b}")
    print(f"ZMK parts: A={zmk_part_a} B={zmk_part_b}")

    # Step 2: Variant for ZPK Part A (A6)
    first_two = zmk_part_b[:2]
    variant_result_a = "{:02x}".format(int("A6", 16) ^ int(first_two, 16))
    varianted_zmk_1 = zmk_part_a + variant_result_a + zmk_part_b[2:]
    print(f"\nVarianted ZMK 1: {varianted_zmk_1}")
    result1 = triple_des_decrypt(zpk_part_a, varianted_zmk_1)
    print(f"Result1: {result1}")

    # Step 3: Variant for ZPK Part B (5A)
    variant_result_b = "{:02x}".format(int("5A", 16) ^ int(first_two, 16))
    varianted_zmk_2 = zmk_part_a + variant_result_b + zmk_part_b[2:]
    print(f"\nVarianted ZMK 2: {varianted_zmk_2}")
    result2 = triple_des_decrypt(zpk_part_b, varianted_zmk_2)
    print(f"Result2: {result2}")

    # Take first 16 chars of each result
    clear_zpk = result1[:16] + result2[:16]
    print(f"\nFinal Clear ZPK: {clear_zpk}")

    return clear_zpk


def generate_kcv(clear_zpk):
    """Generate Key Check Value for the clear ZPK."""
    print(f"\nGenerating KCV for key: {clear_zpk}")

    if len(clear_zpk) != 32:
        raise ValueError(f"Invalid key length: {len(clear_zpk)}, expected 32")

    # Convert hex string to bytes
    key_bytes = binascii.unhexlify(clear_zpk)
    zeros = b"\x00" * 8

    # Create cipher and encrypt
    cipher = DES3.new(key_bytes, DES3.MODE_ECB)
    encrypted = cipher.encrypt(zeros)

    # Return first 6 characters (3 bytes) of encrypted result
    kcv = binascii.hexlify(encrypted).decode()[:6]
    print(f"KCV (first 3 bytes): {kcv}")
    return kcv


if __name__ == "__main__":
    # Test inputs
    encrypted_zpk = "98EEA376A14DF58578E7D77585512DF3"
    encrypted_zmk = "63E4880A2D502DD8E835C68DD8061BBB"
    expected_kcv = "230BEF"

    print("Input Values:")
    print(f"Encrypted ZPK: {encrypted_zpk}")
    print(f"Encrypted ZMK: {encrypted_zmk}")
    print(f"Expected KCV: {expected_kcv}")

    # Compute Clear ZPK
    clear_zpk = compute_clear_zpk(encrypted_zpk, encrypted_zmk)
    print("\nComputed Clear ZPK:", clear_zpk)

    # Generate and verify KCV
    generated_kcv = generate_kcv(clear_zpk)
    print(f"\nKCV Verification:")
    print(f"Expected:  {expected_kcv}")
    print(f"Generated: {generated_kcv.upper()}")

    if generated_kcv.upper() == expected_kcv.upper():
        print("KCV matches!")
    else:
        print("KCV mismatch!")
