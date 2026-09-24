allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}
// Pluginy (np. onnxruntime) bywają zbudowane na android-33, a ich androidx
// wymagają compileSdk >= 34 — wymuszamy nowszy dla modułów bibliotecznych.
// Refleksja celowo: AGP 9 zmienił DSL (BaseExtension usunięty, arity
// CommonExtension zmienne między wersjami).
fun Project.bumpCompileSdk() {
    val androidExt = extensions.findByName("android") ?: return
    val methods = androidExt.javaClass.methods
    val getter = methods.firstOrNull { it.name == "getCompileSdk" && it.parameterCount == 0 }
    val setter = methods.firstOrNull { it.name == "setCompileSdk" && it.parameterCount == 1 }
    if (getter != null && setter != null) {
        val current = getter.invoke(androidExt) as? Int ?: 0
        if (current in 1..33) {
            setter.invoke(androidExt, 36)
            logger.info("compileSdk $current -> 36 dla $name")
        }
    }
}
subprojects {
    if (state.executed) {
        bumpCompileSdk()
    } else {
        afterEvaluate { bumpCompileSdk() }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
