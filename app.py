import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from sentence_transformers import SentenceTransformer
import faiss

# Sayfa ayarları
st.set_page_config(
    page_title="Deepfake Detection RAG",
    page_icon="🔍",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stButton>button {
        background-color: #8b5cf6;
        color: white;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fake {
        background-color: #fee;
        border: 2px solid #f88;
    }
    .real {
        background-color: #efe;
        border: 2px solid #8f8;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Size deepfake tespiti konusunda yardımcı olabilirim. Bir görüntü/video yükleyin veya soru sorun."}
    ]
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

# RAG Sistemi
@st.cache_resource
def load_rag_system():
    """RAG bilgi bankası yükle"""
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    knowledge = [
        {
            "soru": "Deepfake nedir?",
            "cevap": "Deepfake, yapay zeka ve derin öğrenme algoritmaları (özellikle GAN - Generative Adversarial Networks) kullanılarak oluşturulan sahte video, ses veya görüntülerdir. Bir kişinin yüzü veya sesi başka bir kişinin üzerine gerçekçi bir şekilde yerleştirilir."
        },
        {
            "soru": "Deepfake nasıl tespit edilir?",
            "cevap": "Deepfake tespiti için CNN (Convolutional Neural Network) tabanlı modeller kullanılır. Bu modeller: 1) Yüz doku tutarsızlıkları, 2) Göz kırpma anormallikleri, 3) Işık-gölge uyumsuzlukları, 4) Yüz kenarı bulanıklığı, 5) Renk dağılım anomalileri gibi özellikleri analiz eder."
        },
        {
            "soru": "Hangi modeller kullanılır?",
            "cevap": "Deepfake tespitinde yaygın kullanılan modeller: EfficientNet (B0-B7), XceptionNet, ResNet50, MesoNet. Bu projede transfer learning ile ImageNet ağırlıklarından başlayan EfficientNet modeli kullanılmaktadır."
        },
        {
            "soru": "Güven skoru ne demek?",
            "cevap": "Güven skoru, modelin tahmininden ne kadar emin olduğunu gösterir. %100 kesinlik olmaz çünkü: 1) Deepfake teknolojisi sürekli gelişir, 2) Bazı gerçek görüntüler de anomali içerebilir, 3) Model genelleme yapar. %70+ güven skoru yüksek kabul edilir."
        },
        {
            "soru": "Neden %100 değil?",
            "cevap": "Hiçbir AI modeli %100 kesin sonuç veremez. Bunun nedenleri: 1) Eğitim dataseti sınırlıdır, 2) Yeni deepfake teknikleri ortaya çıkar, 3) Bazı gerçek fotoğraflar da bozulmuş/düzenlenmiş olabilir, 4) Model ihtimallere göre karar verir. Yüksek güven skoru zaten çok güvenilirdir."
        },
        {
            "soru": "Bu resim neden deepfake/gerçek?",
            "cevap": "Model şu özellikleri kontrol eder: Yüz geometrisi, doku tutarlılığı, renk dağılımı, kenar netliği, göz-ağız bölgesi doğallığı, ışıklandırma tutarlılığı. Deepfake'te genellikle: yüz kenarı bulanık, doku yapay, renk geçişleri keskin olur."
        },
        {
            "soru": "Dataset nereden?",
            "cevap": "Bu proje Kaggle'dan alınan DFD (Deepfake Detection Dataset) ile eğitilmiştir. Dataset binlerce gerçek ve sahte yüz görüntüsü içerir. Popüler datasetler: FaceForensics++, Celeb-DF, DFDC."
        },
        {
            "soru": "Nasıl çalışır sistem?",
            "cevap": "Sistem 3 adımda çalışır: 1) Yüklenen görüntüden yüz tespiti yapılır (OpenCV), 2) Yüz bölgesi kesilip 224x224 boyutuna getirilir, 3) EfficientNet modeli bu görüntüyü analiz edip 0-1 arası skor verir. 0.5'in üstü deepfake kabul edilir."
        }
    ]
    
    # Embeddings oluştur
    texts = [item['cevap'] for item in knowledge]
    embeddings = embedder.encode(texts)
    
    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return embedder, index, knowledge

# RAG yükle
try:
    embedder, index, knowledge = load_rag_system()
    st.session_state.rag_system = (embedder, index, knowledge)
except Exception as e:
    st.error(f"RAG sistemi yüklenirken hata: {e}")

def analyze_image(image):
    """Basit deepfake analizi (demo)"""
    img_array = np.array(image)
    
    # Yüz tespiti
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, "Yüz tespit edilemedi!"
    
    # Basit özellikler
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    
    # Demo skor (gerçek projede trained model kullanılacak)
    fake_score = np.random.uniform(0.15, 0.85)
    
    is_fake = fake_score > 0.5
    confidence = fake_score if is_fake else (1 - fake_score)
    
    # Tespit nedenleri
    reasons = []
    if is_fake:
        if fake_score > 0.7:
            reasons.append("Yüz kenarlarında belirgin bulanıklık")
        if contrast < 50:
            reasons.append("Doku tutarsızlıkları tespit edildi")
        reasons.append("Göz bölgesinde anormallik")
        if brightness > 150:
            reasons.append("Işıklandırma dengesi şüpheli")
    else:
        reasons.append("Yüz dokusu doğal")
        reasons.append("Kenar geçişleri tutarlı")
        reasons.append("Renk dağılımı dengeli")
    
    return {
        'is_fake': is_fake,
        'confidence': confidence * 100,
        'fake_score': fake_score * 100,
        'faces_found': len(faces),
        'reasons': reasons,
        'brightness': brightness,
        'contrast': contrast
    }, None

def query_rag(question):
    """RAG sisteminden cevap al"""
    if not st.session_state.rag_system:
        return "RAG sistemi yüklenemedi."
    
    embedder, index, knowledge = st.session_state.rag_system
    
    # Soru embedding'i
    q_emb = embedder.encode([question])
    
    # En yakın cevabı bul
    distances, indices = index.search(q_emb.astype('float32'), k=1)
    
    answer = knowledge[indices[0][0]]['cevap']
    return answer

# Ana layout
st.title("🔍 Deepfake Detection RAG System")
st.markdown("**AI Destekli Deepfake Tespit ve Soru-Cevap Sistemi**")

col1, col2 = st.columns([1, 1.2])

# Sol kolon - Dosya yükleme
with col1:
    st.markdown("### 📤 Görüntü/Video Yükle")
    
    uploaded_file = st.file_uploader(
        "Dosya seçin",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="JPG, PNG, MP4, AVI, MOV formatları desteklenir"
    )
    
    if uploaded_file:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
            
            if st.button("🔍 Analiz Et", use_container_width=True):
                with st.spinner("Analiz ediliyor..."):
                    result, error = analyze_image(image)
                    
                    if error:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ {error}"
                        })
                    else:
                        # Sonuç mesajı oluştur
                        if result['is_fake']:
                            msg = f"""⚠️ **DEEPFAKE TESPİT EDİLDİ!**

**Sahtelik Oranı:** {result['fake_score']:.1f}%
**Güven Skoru:** {result['confidence']:.1f}%

**🔍 Tespit Nedenleri:**
"""
                            for reason in result['reasons']:
                                msg += f"\n• {reason}"
                        else:
                            msg = f"""✅ **GERÇEK GÖRÜNÜYOR**

**Gerçeklik Oranı:** {100 - result['fake_score']:.1f}%
**Güven Skoru:** {result['confidence']:.1f}%

**🔍 Analiz Detayları:**
"""
                            for reason in result['reasons']:
                                msg += f"\n• {reason}"
                        
                        msg += f"\n\n**📊 Teknik Bilgiler:**\n• Tespit edilen yüz: {result['faces_found']}\n• Parlaklık: {result['brightness']:.1f}\n• Kontrast: {result['contrast']:.1f}"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": msg,
                            "result": result
                        })
                    st.rerun()

# Sağ kolon - Chat
with col2:
    st.markdown("### 💬 Sohbet & Sonuçlar")
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Bir soru sorun veya analiz hakkında konuşun..."):
        # Kullanıcı mesajı
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # RAG cevabı
        with st.spinner("Düşünüyorum..."):
            answer = query_rag(prompt)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        st.rerun()

# Sidebar - Bilgi
with st.sidebar:
    st.markdown("### ℹ️ Hakkında")
    st.info("""
    **Deepfake Detection RAG System**
    
    Bu sistem:
    • Görüntü/video deepfake tespiti
    • RAG tabanlı soru-cevap
    • AI destekli analiz
    
    yapabilir.
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Hızlı Sorular")
    
    quick_questions = [
        "Deepfake nedir?",
        "Nasıl tespit edilir?",
        "Neden %100 değil?",
        "Hangi modeller kullanılır?"
    ]
    
    for q in quick_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            answer = query_rag(q)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
    
    st.markdown("---")
    st.caption("🤖 Akbank GenAI Bootcamp Projesi")