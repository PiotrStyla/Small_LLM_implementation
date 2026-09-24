# Plugin ML Kit wspomina opcjonalne skrypty, ale aplikacja używa wyłącznie
# dołączonego modelu Latin. Brak klas pozostałych skryptów jest zamierzony.
-dontwarn com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions$Builder
-dontwarn com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
-dontwarn com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions$Builder
-dontwarn com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
-dontwarn com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions$Builder
-dontwarn com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
-dontwarn com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions$Builder
-dontwarn com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions

# ML Kit tworzy rejestratory i logger wizji przez komponenty pośrednie.
# R8 9 usuwa/optymalizuje konstruktor zzmj; OCR działa w debug, a release
# zgłasza NPE w com.google.android.gms.internal.mlkit_vision_common.zzmj.
-keep class com.google.android.gms.internal.mlkit_vision_common.** { *; }
-keep class * implements com.google.firebase.components.ComponentRegistrar {
    public <init>();
}
