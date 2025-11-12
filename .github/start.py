import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_parking_spots(image_path, display=True):
    image = cv2.imread(image_path)
    if image is None:
        print("Nu s-a putut încărca imaginea.")
        return None

    original = image.copy()
    h, w = image.shape[:2]

    # Extindere ROI
    crop_y1 = int(h * 0.20)
    crop_y2 = int(h * 0.80)
    cropped_image = image[crop_y1:crop_y2, :]
   
    # Conversie la Grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicare threshold pentru a detecta marcajele albe
    # Pixelii cu valori peste 160 sunt considerați albi (marcaje)
    _, color_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
   
    # Morfologie (Curățare)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Muchii Canny
    edges = cv2.Canny(mask, 50, 150)
   
    # --- A. DETECTIA LINIILOR VERTICALE ---
    # Linii Hough (Praguri Relaxate)
    lines_v = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=30, minLineLength=30, maxLineGap=10)

    # Filtrare și Grupare Linii Verticale
    vertical_lines = []
    if lines_v is not None:
        for (x1, y1, x2, y2) in lines_v[:, 0]:
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle = abs(np.degrees(angle_rad))
           
            if 75 < angle < 105:
                vertical_lines.append((x1, y1, x2, y2))
               
    def merge_lines(lines, threshold=60):
        if not lines: return []
        lines = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)
        merged = [lines[0]]
        for l in lines[1:]:
            prev_x = np.mean([merged[-1][0], merged[-1][2]])
            curr_x = np.mean([l[0], l[2]])
            if abs(curr_x - prev_x) > threshold:
                merged.append(l)
        return merged

    vertical_lines = merge_lines(vertical_lines, threshold=60)
    vertical_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
   
    # Numărul de locuri pe UN SINGUR RÂND
    num_spots_one_row = max(len(vertical_lines) - 1, 0)
   
    # --- B. DETECTIA LINIEI ORIZONTALE CENTRALE ---
    # Praguri mai relaxate pentru a detecta linia orizontală
    lines_h = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=40, minLineLength=int(w * 0.5), maxLineGap=15)
   
    horizontal_line = None
    if lines_h is not None:
        for (x1, y1, x2, y2) in lines_h[:, 0]:
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle = abs(np.degrees(angle_rad))
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
           
            # Filtrare pentru linii aproape orizontale și suficient de lungi
            if angle < 15 or angle > 165:
                if horizontal_line is None or line_length > np.sqrt((horizontal_line[2] - horizontal_line[0])**2 + (horizontal_line[3] - horizontal_line[1])**2):
                    horizontal_line = (x1, y1, x2, y2)

    # Calculul Final - Presupunem întotdeauna 2 rânduri dacă avem linii verticale
    # În majoritatea parcărilor cu acest tip de marcaj există 2 rânduri
    if num_spots_one_row > 0:
        if horizontal_line is not None:
            num_spots_total = num_spots_one_row * 2
            num_rows_detected = 2
            print("   Linie orizontală detectată - confirmare 2 rânduri")
        else:
            # Dacă nu detectăm linia dar avem linii verticale, probabil sunt tot 2 rânduri
            num_spots_total = num_spots_one_row * 2
            num_rows_detected = 2
            print("    Linie orizontală NU detectată - se presupun 2 rânduri")
    else:
        num_spots_total = 0
        num_rows_detected = 0
       
    # Desenare rezultate
    result = original.copy()
   
    # Linii Verzi Verticale (segmente separate)
    center_y_in_cropped = cropped_image.shape[0] // 2
    line_segment_length = int(cropped_image.shape[0] * 0.7)
    y_offset = line_segment_length // 2
   
    # Segmentul superior
    start_y_upper = crop_y1 + center_y_in_cropped - y_offset
    end_y_upper = crop_y1 + center_y_in_cropped
    # Segmentul inferior
    start_y_lower = crop_y1 + center_y_in_cropped
    end_y_lower = crop_y1 + center_y_in_cropped + y_offset

    for (x1, y1, x2, y2) in vertical_lines:
        line_x_pos = int(np.mean([x1, x2]))
       
        # Desenăm segmentul superior (VERDE)
        cv2.line(result, (line_x_pos, start_y_upper), (line_x_pos, end_y_upper), (0, 255, 0), 3)
        # Desenăm segmentul inferior (VERDE)
        cv2.line(result, (line_x_pos, start_y_lower), (line_x_pos, end_y_lower), (0, 255, 0), 3)

    # Linia Orizontală Centrală (ALBASTRU în BGR)
    if horizontal_line is not None:
        (hx1, hy1, hx2, hy2) = horizontal_line
        cv2.line(result, (hx1, hy1 + crop_y1), (hx2, hy2 + crop_y1), (255, 0, 0), 3)

    # Marcăm locurile numerotate pentru ambele rânduri
    if num_spots_one_row > 0:
        y_ref_upper = crop_y1 + center_y_in_cropped - int(cropped_image.shape[0] * 0.25)
        y_ref_lower = crop_y1 + center_y_in_cropped + int(cropped_image.shape[0] * 0.25)
       
        for i in range(len(vertical_lines) - 1):
            x_curr = int(np.mean([vertical_lines[i][0], vertical_lines[i][2]]))
            x_next = int(np.mean([vertical_lines[i + 1][0], vertical_lines[i + 1][2]]))
            center_x = int((x_curr + x_next) / 2)
           
            # Numerotare Rând Superior (roșu)
            cv2.putText(result, f"{i+1}", (center_x - 10, y_ref_upper),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
           
            # Numerotare Rând Inferior (albastru)
            cv2.putText(result, f"{num_spots_one_row + i+1}", (center_x - 10, y_ref_lower),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    print("=" * 60)
    print(f" Imagine: {image_path}")
    print(f"   ├─ Linii verticale detectate (Separatoare): {len(vertical_lines)}")
    print(f"   ├─ Locuri pe un rând: {num_spots_one_row}")
    print(f"   ├─ Rânduri de parcare detectate: {num_rows_detected}")
    print(f"   └─ Locuri estimate TOTAL: {num_spots_total}")
    print("=" * 60)

    if display:
        plt.figure(figsize=(18, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Imagine originală")
        plt.axis("off")
       
        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap="gray")
        plt.title("Grayscale")
        plt.axis("off")
       
        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap="gray")
        plt.title("Binarizare & Curățare")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(edges, cmap="gray")
        plt.title("Muchii (Canny)")
        plt.axis("off")
       
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f" Locuri detectate: {num_spots_total} ({num_rows_detected} rând{'uri' if num_rows_detected > 1 else ''})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return num_spots_total, result


if __name__ == "__main__":
    image_path = "parcare1.jpg"
    count_parking_spots(image_path)