class Ionic < Formula
  desc "Statically-typed compiled language for data science and AI workloads"
  homepage "https://github.com/henrytunguz/ionic"
  version "v0.1.0"
  license "MIT"

  on_arm do
    url "https://github.com/henrytunguz/ionic/releases/download/#{version}/ionic-#{version}-aarch64-apple-darwin.tar.gz"
    sha256 "PLACEHOLDER_ARM64_SHA256"
  end

  on_intel do
    url "https://github.com/henrytunguz/ionic/releases/download/#{version}/ionic-#{version}-x86_64-apple-darwin.tar.gz"
    sha256 "PLACEHOLDER_X86_SHA256"
  end

  def install
    bin.install "ionic"
    pkgshare.install "lib"
    doc.install "README.md"
  end

  def caveats
    <<~EOS
      The Ionic standard library is installed at:
        #{opt_pkgshare}/lib

      For ONNX model inference, install ONNX Runtime:
        brew install onnxruntime

      For GGUF / LLM inference, install llama.cpp:
        brew install llama.cpp
    EOS
  end

  test do
    (testpath/"hello.ionic").write <<~EOF
      fn main() -> int64 {
        print("hello from ionic");
        return 0;
      }
    EOF
    system bin/"ionic", "hello.ionic", "-o", testpath/"hello"
    assert_equal "hello from ionic", shell_output("#{testpath}/hello").strip
  end
end
