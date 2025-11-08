import cv2
import numpy as np
import matplotlib.pyplot as plt


# 🧩 Funcție pentru combinarea liniilor apropiate (aceeași marcaj)
def merge_similar_lines(lines, orientation='vertical', threshold=20):
    """
    Combină liniile apropiate (care reprezintă același marcaj)
    Args:
        lines: listă de segmente [x1, y1, x2, y2]
        orientation: 'vertical' sau 'horizontal'
        threshold: distanța maximă (în pixeli) pentru a considera două linii identice
    """
    if not lines:
        return []

    merged = []
    if orientation == 'vertical':
        # Sortăm după poziția medie X
        lines = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)
        group = [lines[0]]
        for l in lines[1:]:
            prev_x = np.mean([group[-1][0], group[-1][2]])
            curr_x = np.mean([l[0], l[2]])
            if abs(curr_x - prev_x) < threshold:
                group.append(l)
            else:
                merged.append(np.mean(group, axis=0).astype(int).tolist())
                group = [l]
        merged.append(np.mean(group, axis=0).astype(int).tolist())

    else:  # orizontale
        lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
        group = [lines[0]]
        for l in lines[1:]:
            prev_y = np.mean([group[-1][1], group[-1][3]])
            curr_y = np.mean([l[1], l[3]])
            if abs(curr_y - prev_y) < threshold:
                group.append(l)
            else:
                merged.append(np.mean(group, axis=0).astype(int).tolist())
                group = [l]
        merged.append(np.mean(group, axis=0).astype(int).tolist())

    return merged


# 🅿️ Funcția principală de detecție a liniilor
def detect_parking_lines(image_path, display=True):
    """
    Detectează marcajele albe din imaginea de parcare

    Args:
        image_path: Calea către imagine
        display: Dacă True, afișează rezultatele

    Returns:
        lines_dict: Dicționar cu liniile detectate (verticale/orizontale)
        result_image: Imagine cu liniile desenate
    """

    # Citire imagine
    image = cv2.imread(image_path)
    if image is None:
        print(f"Eroare: Nu s-a putut încărca imaginea {image_path}")
        return None, None

    original = image.copy()

    # Conversie la grayscale și blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # filtrul gaussian ne ajuta sa eliminam detaliile inutile
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectarea marginilor folosind operatorul Canny
    edges = cv2.Canny(blurred, 50, 200, apertureSize=3) # apertureSize indică dimensiunea kernel-ului pentru operatorul Sobel

    # Detectare linii cu Hough Transform (parametri ajustați)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,       # prag mai mare -> mai puține segmente
        minLineLength=80,   # lungime minimă mai mare
        maxLineGap=20       # unește segmente apropiate
    )

    result = original.copy()
    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Clasificare linie
            if 80 <= angle <= 100:
                vertical_lines.append(line[0])
            elif angle <= 15 or angle >= 165:
                horizontal_lines.append(line[0])

    # 🔹 Curățare linii duplicate / apropiate
    vertical_lines = merge_similar_lines(vertical_lines, 'vertical', threshold=25)
    horizontal_lines = merge_similar_lines(horizontal_lines, 'horizontal', threshold=25)

    # 🔹 Desenare pe imagine
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)  # verde - verticale
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)  # roșu - orizontale

    # Afișare rezultate
    if display:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Imaginea Originală')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Detectare Margini (Canny)')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Linii Detectate\nVerticale: {len(vertical_lines)} | Orizontale: {len(horizontal_lines)}')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        vert_img = original.copy()
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(vert_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(vert_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Linii Verticale ({len(vertical_lines)})')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        horiz_img = original.copy()
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(horiz_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.imshow(cv2.cvtColor(horiz_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Linii Orizontale ({len(horizontal_lines)})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Statistici
    print(f"\n📊 Statistici pentru {image_path}:")
    print(f"   ├─ Linii verticale detectate: {len(vertical_lines)}")
    print(f"   ├─ Linii orizontale detectate: {len(horizontal_lines)}")
    print(f"   └─ Total linii: {len(vertical_lines) + len(horizontal_lines)}")

    return {
        'vertical': vertical_lines,
        'horizontal': horizontal_lines,
        'all': lines
    }, result



def detect_parking_spots(image_path):
    """
    Detectează locurile de parcare bazat pe intersecțiile liniilor
    """
    lines_dict, result = detect_parking_lines(image_path, display=False)

    if lines_dict is None:
        return None

    vertical_lines = lines_dict['vertical']
    horizontal_lines = lines_dict['horizontal']

    # Sortare linii verticale după coordonata x
    vertical_lines_sorted = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)

    # Calculare număr locuri de parcare
    num_parking_spots = len(vertical_lines_sorted) - 1 if len(vertical_lines_sorted) > 1 else 0

    print(f"\n🅿  Locuri de parcare detectate: {num_parking_spots}")

    return num_parking_spots


# Utilizare
if __name__ == "__main__":
    # Detectare pentru prima imagine (fără mașini)
    print("=" * 60)
    print("IMAGINE 1 - Parcare goală")
    print("=" * 60)
    lines1, result1 = detect_parking_lines('parcare_libera1.jpg')
    spots1 = detect_parking_spots('parcare_libera1.jpg')
    """""
    print("\n" + "=" * 60)
    print("IMAGINE 2 - Parcare cu mașini")
    print("=" * 60)
    lines2, result2 = detect_parking_lines('parking_occupied.jpg')
    spots2 = detect_parking_spots('parking_occupied.jpg')
    """
    # Salvare rezultate
    if result1 is not None:
        cv2.imwrite('parking_empty_detected.jpg', result1)
        print("\n✅ Rezultat salvat: parking_empty_detected.jpg")
    """""
    if result2 is not None:
        cv2.imwrite('parking_occupied_detected.jpg', result2)
        print("✅ Rezultat salvat: parking_occupied_detected.jpg")"""





