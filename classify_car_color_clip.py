import os
import torch
import clip
from PIL import Image
import shutil
from tqdm import tqdm
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


renkler = ["WHITE", "GRAY", "SILVER", "BROWN", "RED", "BLUE", "YELLOW", "ORANGE", "GREEN","BLACK"]


hedef_klasor = r"G:\Renk_Marka"
os.makedirs(hedef_klasor, exist_ok=True)

for renk in renkler:
    os.makedirs(os.path.join(hedef_klasor, renk), exist_ok=True)

def resmi_siniflandir(resim_yolu):
    image = preprocess(Image.open(resim_yolu)).unsqueeze(0).to(device)
    
    # Renk sınıfları için metin tanımlama
    renk_metinleri = [f"a {renk} car" for renk in renkler]
    text = clip.tokenize(renk_metinleri).to(device)
    
    # Resim ve metin özelliklerini hesaplama
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
    # Özellikleri normalize etme
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Benzerlik skorlarını hesaplama
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # En yüksek benzerlik skoruna sahip rengi seçme
    en_yuksek_indeks = similarity[0].argmax().item()
    esik_degeri = 0.6
    en_yuksek_skor = similarity[0].max().item()
    if en_yuksek_skor < esik_degeri:
        return "belirsiz"
    
    return renkler[en_yuksek_indeks]


def veri_setini_islemek(veri_seti_klasoru):
    for marka_klasoru in os.listdir(veri_seti_klasoru):
        marka_yolu = os.path.join(veri_seti_klasoru, marka_klasoru)
        
        if os.path.isdir(marka_yolu):
            for resim_dosyasi in tqdm(os.listdir(marka_yolu), desc=f"İşleniyor: {marka_klasoru}"):
                resim_yolu = os.path.join(marka_yolu, resim_dosyasi)
                
                if resim_yolu.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Resmin rengini tahmin et
                        renk = resmi_siniflandir(resim_yolu)
                        if renk == 'belirsiz':
                            continue
                        
                        # Hedef dosya adını oluştur (marka_model_id.jpg formatında)
                        hedef_dosya_adi = f"{marka_klasoru}_{os.path.splitext(resim_dosyasi)[0]}.jpg"
                        hedef_yol = os.path.join(hedef_klasor, renk, hedef_dosya_adi)
                        
                        # Resmi hedef klasöre kopyala
                        shutil.copy2(resim_yolu, hedef_yol)
                        
                    except Exception as e:
                        print(f"Hata: {resim_yolu} işlenirken bir sorun oluştu - {e}")



if __name__ == "__main__":
    veri_seti_klasoru = r"G:\PaddleClas_Dataset_Root_MarkaNew"  # Veri setinizin bulunduğu klasör
    veri_setini_islemek(veri_seti_klasoru)
    print("İşlem tamamlandı!")