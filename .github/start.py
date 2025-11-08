import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_parking_lines(image_path, display=True):
    """
    Detectează marcajele albe din imaginea de parcare
    
    Args:
        image_path: Calea către imagine
        display: Dacă True, afișează rezultatele
    
    Returns:
        lines: Lista cu liniile detectate
        result_image: Imaginea cu liniile desenate
    """
    
    # Citire imagine
    image = cv2.imread(image_path)
    if image is None:
        print(f"Eroare: Nu s-a putut încărca imaginea {image_path}")
        return None, None
    
    original = image.copy()
    
    # Conversie la grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicare blur pentru reducerea zgomotului
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectare margini cu Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detectare linii cu Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,              # Rezoluție în pixeli
        theta=np.pi/180,    # Rezoluție în radiani
        threshold=50,       # Număr minim de intersecții
        minLineLength=30,   # Lungime minimă linie
        maxLineGap=10       # Gap maxim între segmente
    )
    
    # Creare imagine rezultat
    result = original.copy()
    
    # Separare linii verticale și orizontale
    vertical_lines = []
    horizontal_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculare unghi
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Clasificare: verticale (aproape de 90°) sau orizontale (aproape de 0/180°)
            if angle > 80 and angle < 100:
                vertical_lines.append(line[0])
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde pentru verticale
            elif angle < 10 or angle > 170:
                horizontal_lines.append(line[0])
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Roșu pentru orizontale
    
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
        
        # Afișare doar linii verticale
        vertical_img = original.copy()
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(vertical_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(vertical_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Linii Verticale ({len(vertical_lines)})')
        plt.axis('off')
        
        # Afișare doar linii orizontale
        horizontal_img = original.copy()
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(horizontal_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(horizontal_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Linii Orizontale ({len(horizontal_lines)})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
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