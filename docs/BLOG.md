# Optimizing Qwen 2.5 0.5B Inference on NVIDIA GPUs: Beating Hugging Face with Bare-Metal CUDA

**Author:** [Amr Gamal (CaptainLevi04)](https://github.com/CaptainLevi04)  
**Model Checkpoint:** [`Amr04/Qwen2.5-0.5B-AWQ-W4A16-tied-embed-asym-hq128`](https://huggingface.co/Amr04/Qwen2.5-0.5B-AWQ-W4A16-tied-embed-asym-hq128)  
**GitHub Repository:** [https://github.com/CaptainLevi04/quic.cu](https://github.com/CaptainLevi04/quic.cu)  
**Date:** November 2026  

---

﻿<!DOCTYPE html><html dir="ltr" lang="en" class="scroll-smooth"> <head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link rel="icon" type="image/svg+xml" href="/favicon.svg"><link rel="canonical" href="https://hdahroug.github.io/optimizing-qwen2.5-0.5b-cuda/"><meta name="generator" content="Astro v5.16.6"><!-- Load Font --><style>@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/eac50eb620c54c10.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:300;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/eac50eb620c54c10.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:400;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/eac50eb620c54c10.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:500;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/eac50eb620c54c10.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:600;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/eac50eb620c54c10.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:700;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/408661687ee6f3ed.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:300;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/408661687ee6f3ed.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:400;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/408661687ee6f3ed.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:500;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/408661687ee6f3ed.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:600;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/408661687ee6f3ed.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:700;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/2213c1e7fbe06cc7.woff") format("woff");font-display:swap;font-weight:300;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/c487dbd9c215141e.woff") format("woff");font-display:swap;font-weight:400;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/0976c79a1e97c08d.woff") format("woff");font-display:swap;font-weight:500;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/3c46d5a1dde2929d.woff") format("woff");font-display:swap;font-weight:600;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/e9aa617e80052f4b.woff") format("woff");font-display:swap;font-weight:700;font-style:italic;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/5056f1a50e5e7744.woff") format("woff");font-display:swap;font-weight:300;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/e734307f8c328338.woff") format("woff");font-display:swap;font-weight:400;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/dfa89044152f7da4.woff") format("woff");font-display:swap;font-weight:500;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/8bbd58b325585200.woff") format("woff");font-display:swap;font-weight:600;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992";src:url("/_astro/fonts/035035a9ddbd14aa.woff") format("woff");font-display:swap;font-weight:700;font-style:normal;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:300;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:400;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:500;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:600;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:700;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:300;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:400;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:500;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:600;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:700;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:300;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:400;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:500;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:600;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:700;font-style:italic;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:300;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:400;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:500;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:600;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}@font-face{font-family:"Google Sans Code-04a186ad63205992 fallback: Courier New";src:local("Courier New");font-display:swap;font-weight:700;font-style:normal;size-adjust:99.9837%;ascent-override:96.6157%;descent-override:28.6047%;}:root{--font-google-sans-code:"Google Sans Code-04a186ad63205992","Google Sans Code-04a186ad63205992 fallback: Courier New",monospace;}</style><style>@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/573727ed382a8c73.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:400;font-style:italic;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/50609c4acdbbc958.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:700;font-style:italic;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/e5e178ad67270bfe.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:400;font-style:normal;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/062c04dbe0fef904.woff2") format("woff2");font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-weight:700;font-style:normal;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/c3bdaf2f9a49cf21.woff") format("woff");font-display:swap;font-weight:400;font-style:italic;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/85dc22427f88c577.woff") format("woff");font-display:swap;font-weight:700;font-style:italic;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/604df18af53afdb9.woff") format("woff");font-display:swap;font-weight:400;font-style:normal;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5";src:url("/_astro/fonts/ae06223db35f1ada.woff") format("woff");font-display:swap;font-weight:700;font-style:normal;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:400;font-style:italic;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:700;font-style:italic;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:400;font-style:normal;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:700;font-style:normal;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:400;font-style:italic;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:700;font-style:italic;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:400;font-style:normal;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}@font-face{font-family:"PT Serif-c1b3c5e980df12c5 fallback: Times New Roman";src:local("Times New Roman");font-display:swap;font-weight:700;font-style:normal;size-adjust:104.8615%;ascent-override:99.083%;descent-override:27.2741%;line-gap-override:0%;}:root{--font-pt-serif:"PT Serif-c1b3c5e980df12c5","PT Serif-c1b3c5e980df12c5 fallback: Times New Roman",serif;}</style><link rel="preload" href="/_astro/fonts/e5e178ad67270bfe.woff2" as="font" type="font/woff2" crossorigin><link rel="preload" href="/_astro/fonts/062c04dbe0fef904.woff2" as="font" type="font/woff2" crossorigin><!-- General Meta Tags --><title>Optimizing Qwen 2.5 0.5B Inference on NVIDIA GPUs: Beating Hugging Face with Bare-Metal CUDA | Amr Gamal</title><meta name="title" content="Optimizing Qwen 2.5 0.5B Inference on NVIDIA GPUs: Beating Hugging Face with Bare-Metal CUDA | Hassan Dahroug"><meta name="description" content="Building QUIC, a bare-metal CUDA serving engine for Qwen 2.5 0.5B from scratch, achieving up to 2.62x higher decode throughput over Hugging Face FP16 on an NVIDIA H100."><meta name="author" content="Dahroug"><link rel="sitemap" href="/sitemap-index.xml"><!-- Open Graph / Facebook --><meta property="og:title" content="Optimizing Qwen 2.5 0.5B Inference on NVIDIA GPUs: Beating Hugging Face with Bare-Metal CUDA | Hassan Dahroug"><meta property="og:description" content="Building QUIC, a bare-metal CUDA serving engine for Qwen 2.5 0.5B from scratch, achieving up to 2.62x higher decode throughput over Hugging Face FP16 on an NVIDIA H100."><meta property="og:url" content="https://hdahroug.github.io/optimizing-qwen2.5-0.5b-cuda/"><meta property="og:image" content="https://hdahroug.github.io/optimizing-qwen2.5-0.5b-cuda/index.png"><!-- Article Published/Modified time --><meta property="article:published_time" content="2026-10-23T23:25:00.000Z"><!-- Twitter --><meta property="twitter:card" content="summary_large_image"><meta property="twitter:url" content="https://hdahroug.github.io/optimizing-qwen2.5-0.5b-cuda/"><meta property="twitter:title" content="Optimizing Qwen 2.5 0.5B Inference on NVIDIA GPUs: Beating Hugging Face with Bare-Metal CUDA | Hassan Dahroug"><meta property="twitter:description" content="Building QUIC, a bare-metal CUDA serving engine for Qwen 2.5 0.5B from scratch, achieving up to 2.62x higher decode throughput over Hugging Face FP16 on an NVIDIA H100."><meta property="twitter:image" content="https://hdahroug.github.io/optimizing-qwen2.5-0.5b-cuda/index.png"><!-- Google JSON-LD Structured data --><script type="application/ld+json">{"@context":"https://schema.org","@type":"BlogPosting","headline":"Optimizing Qwen 2.5 0.5B Inference on NVIDIA GPUs: Beating Hugging Face with Bare-Metal CUDA | Hassan Dahroug","image":"https://hdahroug.github.io/optimizing-qwen2.5-0.5b-cuda/index.png","datePublished":"2026-10-23T23:25:00.000Z","author":[{"@type":"Person","name":"Dahroug","url":"https://hdahroug.github.io/"}]}</script><!-- Enable RSS feed auto-discovery  --><!-- https://docs.astro.build/en/recipes/rss/#enabling-rss-feed-auto-discovery --><link rel="alternate" type="application/rss+xml" title="Hassan Dahroug" href="https://hdahroug.github.io/rss.xml"><meta name="theme-color" content=""><meta name="google-site-verification" content="Ap5bb66JIeOrmzMQt4h9akaR9l0GoOiwcrxrHPknm58"><meta name="astro-view-transitions-enabled" content="true"><meta name="astro-view-transitions-fallback" content="animate"><script type="module" src="/_astro/ClientRouter.astro_astro_type_script_index_0_lang.QW52Ox2j.js"></script><!-- Minimal inline script to prevent FOUC - sets theme immediately --><script>
      (function () {
        const initialColorScheme = ""; // "light" | "dark"
        const currentTheme = localStorage.getItem("theme");

        function getPreferTheme() {
          if (currentTheme) return currentTheme;
          if (initialColorScheme) return initialColorScheme;
          return window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light";
        }

        const themeValue = getPreferTheme();

        // Set theme immediately to prevent flash
        document.firstElementChild?.setAttribute("data-theme", themeValue);

        // Export minimal API for external script
        window.theme = {
          themeValue: themeValue,
          getTheme: () => window.theme.themeValue,
          setTheme: val => {
            window.theme.themeValue = val;
          },
        };
      })();
    </script><style>:where([data-astro-image]){object-fit:var(--fit);object-position:var(--pos);height:auto}:where([data-astro-image=full-width]){width:100%}:where([data-astro-image=constrained]){max-width:100%}
</style>
<link rel="stylesheet" href="style.css">
<style>@keyframes astroFadeInOut{0%{opacity:1}to{opacity:0}}@keyframes astroFadeIn{0%{opacity:0;mix-blend-mode:plus-lighter}to{opacity:1;mix-blend-mode:plus-lighter}}@keyframes astroFadeOut{0%{opacity:1;mix-blend-mode:plus-lighter}to{opacity:0;mix-blend-mode:plus-lighter}}@keyframes astroSlideFromRight{0%{transform:translate(100%)}}@keyframes astroSlideFromLeft{0%{transform:translate(-100%)}}@keyframes astroSlideToRight{to{transform:translate(100%)}}@keyframes astroSlideToLeft{to{transform:translate(-100%)}}@media (prefers-reduced-motion){::view-transition-group(*),::view-transition-old(*),::view-transition-new(*){animation:none!important}[data-astro-transition-scope]{animation:none!important}}
</style><style>[data-astro-transition-scope="astro-yb6n2ciq-1"] { view-transition-name: optimizing-qwen-2-5-0-5b-inference-on-nvidia-gpus_3a-beating-hugging-face-with-bare-metal-cuda; }@layer astro { ::view-transition-old(optimizing-qwen-2-5-0-5b-inference-on-nvidia-gpus_3a-beating-hugging-face-with-bare-metal-cuda) { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeOut; }::view-transition-new(optimizing-qwen-2-5-0-5b-inference-on-nvidia-gpus_3a-beating-hugging-face-with-bare-metal-cuda) { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeIn; }[data-astro-transition=back]::view-transition-old(optimizing-qwen-2-5-0-5b-inference-on-nvidia-gpus_3a-beating-hugging-face-with-bare-metal-cuda) { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeOut; }[data-astro-transition=back]::view-transition-new(optimizing-qwen-2-5-0-5b-inference-on-nvidia-gpus_3a-beating-hugging-face-with-bare-metal-cuda) { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeIn; } }[data-astro-transition-fallback="old"] [data-astro-transition-scope="astro-yb6n2ciq-1"],
			[data-astro-transition-fallback="old"][data-astro-transition-scope="astro-yb6n2ciq-1"] { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeOut; }[data-astro-transition-fallback="new"] [data-astro-transition-scope="astro-yb6n2ciq-1"],
			[data-astro-transition-fallback="new"][data-astro-transition-scope="astro-yb6n2ciq-1"] { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeIn; }[data-astro-transition=back][data-astro-transition-fallback="old"] [data-astro-transition-scope="astro-yb6n2ciq-1"],
			[data-astro-transition=back][data-astro-transition-fallback="old"][data-astro-transition-scope="astro-yb6n2ciq-1"] { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeOut; }[data-astro-transition=back][data-astro-transition-fallback="new"] [data-astro-transition-scope="astro-yb6n2ciq-1"],
			[data-astro-transition=back][data-astro-transition-fallback="new"][data-astro-transition-scope="astro-yb6n2ciq-1"] { 
	animation-duration: 180ms;
	animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
	animation-fill-mode: both;
	animation-name: astroFadeIn; }</style>
  <!-- Google Fonts & KaTeX for pristine typography and math -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=PT+Serif:ital,wght@0,400;0,700;1,400;1,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\[', right: '\\]', display: true}, {left: '$', right: '$', display: false}, {left: '\\(', right: '\\)', display: false}]});"></script>
  <style>
    :root {
      --background: #fdfdfd !important;
      --foreground: #111111 !important;
      --accent: #1f53a3 !important;
      --font-pt-serif: 'PT Serif', Georgia, serif;
      --font-google-sans-code: 'JetBrains Mono', monospace;
    }
    body {
      background-color: #fdfdfd !important;
      color: #111111 !important;
      font-family: var(--font-pt-serif), Georgia, serif !important;
    }
    /* Ensure clean typography matching original paper */
    .app-prose {
      max-width: 68ch !important;
      margin-left: auto;
      margin-right: auto;
    }
    .app-prose table {
      width: 100% !important;
      margin: 1.5rem 0;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    .app-prose th, .app-prose td {
      border: 1px solid #e5e7eb;
      padding: 0.6rem 0.8rem;
    }
    .app-prose th {
      background-color: #f9fafb;
      font-weight: 600;
    }
    .app-prose tr:nth-child(even) td {
      background-color: #fafafa;
    }
    .app-prose img {
      max-width: 100%;
      height: auto;
      display: block;
      margin: 1.75rem auto;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
    }
    .img-caption {
      font-size: 0.85rem;
      color: #6b7280;
      text-align: center;
      margin-top: -1rem;
      margin-bottom: 1.75rem;
      font-style: italic;
    }
    blockquote {
      border-left: 4px solid #d1d5db !important;
      padding-left: 1rem !important;
      margin: 1.5rem 0 !important;
      color: #374151 !important;
      font-style: italic !important;
    }
    pre {
      background-color: #1e1e1e !important;
      color: #d4d4d4 !important;
      padding: 1rem !important;
      border-radius: 6px !important;
      overflow-x: auto !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 0.85rem !important;
      line-height: 1.5 !important;
    }
    pre code {
      color: inherit !important;
      background: none !important;
      border: none !important;
      padding: 0 !important;
    }
    code:not(pre code) {
      background-color: #f3f4f6;
      color: #1f53a3;
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
      font-size: 0.875em;
      font-family: 'JetBrains Mono', monospace;
    }
  </style>

</head> <body>  [ <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="inline-block size-6 rtl:rotate-180"><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path d="M15 6l-6 6l6 6" /></svg>Amr Gamal ](/)<script type="module">function e(){const t=document.querySelector("#back-button");t&&(t.href="/")}document.addEventListener("astro:page-load",e);e();</script>  #  Optimizing Qwen 2.5 0.5B Inference on NVIDIA GPUs: Beating Hugging Face with Bare-Metal CUDA 

    Nov 2026  |    **TL;DR**

The code, kernels, and benchmarks are open-source on GitHub at [CaptainLevi04/quic.cu](https://github.com/CaptainLevi04/quic.cu). The pre-quantized 4-bit model checkpoint is published on Hugging Face at [Amr04/Qwen2.5-0.5B-AWQ-W4A16](https://huggingface.co/Amr04/Qwen2.5-0.5B-AWQ-W4A16-tied-embed-asym-hq128).

After spending time writing isolated CUDA kernels, I wanted to understand what actually happens when you chain everything together into a real LLM inference engine. Hugging Face eager execution is great for prototyping, but when you open a profiler trace on a modern GPU, you see thousands of micro-kernel launches, huge memory thrashing, and massive idle bubbles.

So I built QUIC, a bare-metal CUDA serving engine for Qwen 2.5 0.5B from scratch, and benchmarked it head-to-head against Hugging Face on an NVIDIA H100 GPU.

The result is a clean C++ and CUDA runtime with hand-written fused kernels for RMSNorm, RoPE, SwiGLU, and Split-K GQA decode attention, wrapped with cuBLASLt GEMMs, native 4-bit AWQ Marlin quantization, and an iteration-level continuous batching scheduler. It consistently beats Hugging Face FP16 by **2.44x to 2.62x higher decode throughput**, reaching up to **1,876 tokens per second**, while also cutting prompt TTFT significantly.

The more interesting findings came when measuring the real bottlenecks:

First, **quantization behaves counter-intuitively on small models**. On a 70B model, 4-bit weights make decode fly. On Qwen 2.5 0.5B, packing weights with AWQ cuts peak VRAM in half from 1,548 MB down to 780 MB, but the ALU overhead of unpacking 4-bit integers on the fly slightly offsets memory bandwidth gains. It is a capacity win for fitting longer contexts, not a latency cheat code.

Second, **attention on big GPUs starves without Split-K**. An H100 has 132 SMs, but Qwen 2.5 0.5B only has 14 query heads. At batch size 1, standard FlashAttention launches only 14 thread blocks, leaving 118 SMs completely dead. Slicing historical context into 64-token chunks wakes up the entire GPU during decode.

Third, **continuous batching delivers free performance**. Scheduling requests at the token iteration level rather than the request level gave a **+15.7 percent serving throughput boost** on identical hardware without touching a single CUDA instruction.

And during verification, **QUIC achieves 100 percent word-for-word identity with Hugging Face** across 100 generated tokens on open-ended prompts, proving you do not have to sacrifice numerical accuracy for speed.

<p>

![NVIDIA H100 Raw Model Benchmark: QUIC Custom CUDA Kernels vs Hugging Face FP16 Decode Throughput and TTFT Latency](images/nvidia-h100-quic-vs-huggingface-latency-throughput.png)
*NVIDIA H100 Raw Model Benchmark: QUIC Custom CUDA Kernels vs Hugging Face FP16 Decode Throughput and TTFT Latency*

</p>
<p>

![NVIDIA H100 Quantization and Batching Benchmark: W4A16 Marlin vs FP16 and Continuous vs Static Scheduling](images/nvidia-h100-awq-marlin-quantization-continuous-batching.png)
*NVIDIA H100 Quantization and Batching Benchmark: W4A16 Marlin vs FP16 and Continuous vs Static Scheduling*

</p>
**Note:** All benchmarks were run on an NVIDIA H100 80GB HBM3 GPU via Modal, sweeping across batch sizes 1, 4, 16 and context lengths from 128 to 2048 tokens.

<blockquote>
Feel free to reach out if you want to build together. My contact info is right at the bottom of this page.

</blockquote>
**Blog Map**

1. [Why Build an LLM Engine from Scratch?](#why-build-an-llm-engine-from-scratch)
2. [Profiling Hugging Face Eager Mode](#profiling-hugging-face-eager-mode)
3. [Qwen 2.5 Architecture and the Bare-Metal Graph](#qwen-25-architecture-and-the-bare-metal-graph)
4. [Writing the Core CUDA Workhorses](#writing-the-core-cuda-workhorses)
5. [The Attention Frontier: Prefill vs Decode](#the-attention-frontier-prefill-vs-decode)
6. [Quantization: The 0.5B Reality Check](#quantization-the-05b-reality-check)
7. [System-Level Serving: Dynamic KV-Cache and Continuous Batching](#system-level-serving-dynamic-kv-cache-and-continuous-batching)
8. [End-to-End Benchmarks, Profiler Deep-Dive, and Accuracy Parity](#end-to-end-benchmarks-profiler-deep-dive-and-accuracy-parity)

<hr>

---

## Why Build an LLM Engine from Scratch?

After spending time optimizing CUDA kernels on GPUs, I wanted to understand what actually happens under the hood in LLM inference. When you run an isolated GEMM or attention kernel, you only see one piece of the puzzle. An actual inference pipeline chains dozens of different operations together across multiple transformer layers, passing tensors back and forth.

I wanted to see where the time actually goes in an end to end generation run, how the memory hierarchy behaves, and how different optimizations affect performance differently. Which optimization affects which metric, and by how much?

---

## Why Qwen 2.5 0.5B?

I chose Qwen 2.5 0.5B because it has a modern architecture and is not overly complicated. It uses Grouped Query Attention with a 7 to 1 ratio, Rotary Position Embeddings, SwiGLU activations, and RMSNorm.

Larger models like Qwen 2.5 7B, 14B, or 72B share almost the exact same architecture, just with larger dimensions and head counts. What you learn on the 0.5B version transfers directly to the bigger models.

If you want a solid visual walkthrough of the underlying transformer attention mechanics before looking at low-level CUDA code, Jay Alammar’s [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) is the best place to start.

The 0.5B size also means a very fast feedback loop and faster development. In FP16, the whole model is roughly 0.98 GB. It downloads in seconds, loads into GPU memory immediately, and tests run in a few seconds. If a custom kernel has a bug, causes race conditions, or outputs NaNs, you catch it right away without waiting through long compilation and load cycles.

---

## The Targets and Metrics

The main target is to beat the Hugging Face inference stack on a production grade GPU and understand how each optimization moves the needle.

To clearly evaluate our engine, we track 3 main performance metrics:

- **TTFT:** Time to first token in milliseconds. This measures the prompt prefill stage, which is compute bound.

- **TPS:** Tokens per second during generation. This measures the token by token decode loop, which is mostly memory bandwidth bound.

- **VRAM:** Peak GPU memory allocated in megabytes. Lower memory usage directly determines how large of a batch size or context length you can serve.

<p>

![Time to First Token TTFT vs Tokens Per Second TPS LLM Inference Tradeoff Curve](images/ttft-vs-tps-llm-inference-tradeoff.png)
*Time to First Token TTFT vs Tokens Per Second TPS LLM Inference Tradeoff Curve*

</p>
Plus 1 accuracy metric:

- **Absolute logits difference:** measuring max and mean divergence against stock Hugging Face FP16 to make sure our fused kernels do not degrade generation quality or drift numerically across layers.

---

## Profiling Hugging Face Eager Mode

Before writing any custom kernels or changing the architecture, we have to know what Hugging Face inference looks like and where it falls short.

Looking at [modeling_qwen2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) in the Hugging Face transformers repository, the eager implementation runs each layer as a sequence of independent PyTorch operations.

When you profile a decode step using torch.profiler, the main issues become obvious right away:

1. **Kernel launch overhead:** each of the 24 layers runs input RMSNorm, QKV projections, RoPE, attention, output projection, post-attention RMSNorm, Gate projection, Up projection, SwiGLU activation, and Down projection. That is over 10 separate kernel launches per layer. Across 24 layers, that is more than 250 kernel launches for one single generated token. When batch size is small, the GPU finishes these tiny kernels so fast that the CPU spends more time dispatching launches than the GPU spends executing them.
2. **Memory traffic on intermediate tensors:** every single operation writes its output back to global GPU memory. For example, SwiGLU computes the gate projection, writes it to memory, computes SiLU on it, writes that intermediate tensor back to memory, loads the up projection, and then multiplies them. The GPU memory bus gets saturated moving intermediate activations that are never needed again.
3. **Framework overhead:** dynamic cache updates, device checks, and tensor layout validations in Python add constant latency to every decode iteration.

If you want to learn how to run torch.profiler and interpret trace timelines like a pro, Quentin Anthony’s [PyTorch Profiling Tutorial](https://github.com/Quentin-Anthony/torch-profiler-tutorial) is an incredible guide that covers how to spot these exact bottlenecks.

### Visualizing the Carnage on the Timeline

To see this with my own eyes, I set up torch.profiler and recorded both Hugging Face eager execution and QUIC on an NVIDIA H100 80GB HBM3 GPU.

I warmed up each model with 1 prefill step and 10 decode steps to reach steady state, and then captured 1 prefill followed by 5 decode steps at Batch Size 1 and Context Length 512.

When you export the trace file `images/hf_trace.json.gz` and drop it into `ui.perfetto.dev`, the reality hits you like a truck.

<table><thead><tr><th align="left">Timeline Metric</th><th align="left">Hugging Face Eager FP16</th><th align="left">QUIC Engine FP16</th><th align="left">The Win</th></tr></thead><tbody><tr><td align="left">**Total Kernel Launches**</td><td align="left">6630</td><td align="left">3695</td><td align="left">**1.79x fewer launches**</td></tr><tr><td align="left">**Kernel Launches Per Step**</td><td align="left">1105.0</td><td align="left">615.8</td><td align="left">**489 launches eliminated per step**</td></tr><tr><td align="left">**Total CUDA Active Compute**</td><td align="left">17.54 ms</td><td align="left">11.14 ms</td><td align="left">**1.57x faster compute**</td></tr><tr><td align="left">**Total Timeline Wall Span**</td><td align="left">185.58 ms</td><td align="left">77.20 ms</td><td align="left">**2.40x speedup**</td></tr><tr><td align="left">**CPU Dispatch Idle Bubbles**</td><td align="left">168.04 ms</td><td align="left">66.05 ms</td><td align="left">**101.98 ms empty gaps eliminated**</td></tr><tr><td align="left">**GPU Stream Duty Cycle**</td><td align="left">9.5 percent</td><td align="left">14.4 percent</td><td align="left">**+5.0 percent continuous utilization**</td></tr><tr><td align="left">**Peak Memory Allocated**</td><td align="left">1633.2 MB</td><td align="left">1147.9 MB</td><td align="left">**485.3 MB memory footprint saved**</td></tr></tbody></table>
<p>

![Torch Profiler Perfetto Timeline Comparison: Hugging Face Eager Dispatch Bubbles vs QUIC Fused CUDA Kernels](images/torch-profiler-perfetto-timeline-huggingface-vs-quic.png)
*Torch Profiler Perfetto Timeline Comparison: Hugging Face Eager Dispatch Bubbles vs QUIC Fused CUDA Kernels*

</p>
<p>

![Perfetto Trace Zoom of Decode Iteration: 1100 Micro-Kernel Launches in PyTorch vs Streamlined CUDA Kernels](images/perfetto-decode-iteration-timeline-kernel-bubbles.png)
*Perfetto Trace Zoom of Decode Iteration: 1100 Micro-Kernel Launches in PyTorch vs Streamlined CUDA Kernels*

</p>
Here is what the timeline looks like when you zoom in on Hugging Face:

1. **The 1100-Launch Storm:** In every single decode iteration, Hugging Face fires over 1100 individual CUDA kernel launches. You see endless slivers of `aten::mul`, `aten::add`, `aten::copy_`, and trivial elementwise casts scattered across the stream.
2. **The 168 Millisecond Void:** The raw CUDA execution time was only 17.54 milliseconds. Yet the total timeline span dragged on for 185.58 milliseconds. That means 168.04 milliseconds, over 90 percent of the entire run, was pure dead air. The GPU was sitting completely idle, waiting for Python bytecode and PyTorch dispatch logic to queue the next micro-kernel. The GPU stream duty cycle was a pathetic 9.5 percent.
3. **Memory Thrashing:** Hugging Face allocated and freed intermediate buffers on nearly every operation, chewing up 1633.2 MB of VRAM for a model that only weighs 988 MB.

When you load `images/quic_trace.json.gz`, the picture changes completely:

- The hundreds of tiny elementwise slivers are gone, replaced by dense, contiguous blocks of fused kernels: `qwen2_rmsnorm_kernel`, cuBLASLt GEMM with fused bias, and `qwen2_rope_kernel`.

- The CPU dispatch bubbles shrink by more than 101 milliseconds.

- Peak VRAM drops to 1147.9 MB without any dynamic allocation churn during decode.

- The entire timeline span collapses from 185.58 ms to 77.20 ms, giving an immediate 2.40x speedup on the timeline.

<hr>

---

## Qwen 2.5 Architecture and the Bare-Metal Graph

To optimize Qwen 2.5 0.5B, we have to look closely at its architecture and how each tensor moves through the model. For the official architectural background and training details, check out the [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115). If you just call transformers from Hugging Face, you inherit all its runtime abstractions without knowing what is happening. We need to look at the dimensions, how attention heads are split, and how weights are stored in memory.

The model parameters come directly from `config.json`:

- Hidden dimension: 896

- Intermediate MLP dimension: 4864

- Query heads: 14 heads, each with dimension 64, giving 14 * 64 = 896

- Key and Value heads: 2 heads, each with dimension 64, giving 2 * 64 = 128

- GQA ratio: 7 query heads share 1 KV head, 14 divided by 2

- Vocab size: 151,936

- RoPE theta: 1,000,000.0

- RMSNorm eps: 1e-6

- Number of hidden layers: 24

<p>

![Qwen 2.5 0.5B Model Architecture Hyperparameters and Layer Configuration Diagram](images/qwen2-5-0-5b-model-architecture-configuration.png)
*Qwen 2.5 0.5B Model Architecture Hyperparameters and Layer Configuration Diagram*

</p>

### Where Do the 494M Parameters Actually Live?

People call it Qwen 0.5B, but calculating the exact parameter count by hand reveals how the weights are actually distributed between embeddings, attention, and the feed-forward network.

If you want a granular breakdown of these calculations and how they map to hardware constraints, [Lecture 01 of the AI Hardware Engineer Roadmap](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/blob/f48da187c409a07fdd70a52c2b9f1568a03310b5/Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20C%20-%20Edge%20AI/Qwen%20Inference%20Optimization/Lecture-01.md) walks through the exact parameter counting and layer math step by step.

Here is the exact manual parameter breakdown:

1. Input Embeddings:

- 151,936 vocab * 896 hidden = 136.14 M parameters
2. Per-Layer Attention:

- Wq and Wo: 2 * 14 heads * 64 head_dim * 896 hidden = 1,605,632

- Wk and Wv: 2 * 2 heads * 64 head_dim * 896 hidden = 229,376

- Attention projections total: around 1.835 M parameters

- Bias terms: 896 * 2 + 2 * 128 = around 2 K parameters
3. Per-Layer Feed-Forward Network:

- Gate, Up, and Down projections: 3 * 4864 intermediate * 896 hidden = 13,074,432, roughly 13.074 M parameters
4. Per-Layer RMSNorms:

- Input norm and post-attention norm: 2 * 896 = 1,792 parameters
5. Total for 24 Layers:

- Each layer total = 1.835 M + 13.074 M = around 14.91 M parameters

- 24 layers * 14.91 M = 357.9 M parameters
6. Final Model Total:

- 136.1 M embeddings + 357.9 M layers + final norm around 896 = **494 M parameters**

<p>

![Qwen 2.5 0.5B Parameter Distribution and Feed-Forward Network Weight Calculations](images/qwen2-5-parameter-distribution-ffn-breakdown.png)
*Qwen 2.5 0.5B Parameter Distribution and Feed-Forward Network Weight Calculations*

</p>

### The Tied Embeddings Memory Saving

Notice that the input embeddings alone account for 136.1 M parameters. In FP16, that is:

151936 * 896 * 2 bytes = 272.2 MB

At the end of the 24 layers, the model needs to project the final hidden states back to vocabulary logits through `lm_head`. That projection also requires a 151,936 by 896 matrix.

In Qwen 2.5 0.5B, `config.tie_word_embeddings` is set to True. The input embedding table and the output projection matrix share the exact same weights in memory.

If you load them separately, you waste 272 MB of GPU VRAM duplicating identical weights. In our engine loader in `src/model.py`, we explicitly tie them right after loading the safetensors:

```
self.lm_head.weight = self.model.embed_tokens.weight
```

### The Real Memory Math: KV Cache and VRAM Overheads

When I was first messing around on my local GTX 1650 with its tiny 4 GB of VRAM, before firing up beefy H100 instances on Modal, I realized you cannot just look at the 988 MB model weights and assume you have 3 GB of free space.

There are two major memory consumers you have to budget for before even launching a kernel.

#### The 450 MB Invisible VRAM Floor

Even before your model processes a single token, three separate runtime components quietly steal almost half a gigabyte of VRAM:

<p>

![GPU VRAM Memory Allocation Breakdown: 450 MB Runtime Floor of CUDA Driver, cuBLAS, and PyTorch Caching Allocator](images/gpu-vram-driver-overhead-and-allocator-floor.png)
*GPU VRAM Memory Allocation Breakdown: 450 MB Runtime Floor of CUDA Driver, cuBLAS, and PyTorch Caching Allocator*

</p>
1. **CUDA Driver Context:** takes around 100 to 150 MB just for hardware page tables, device runtime state, and CUDA stream queues.
2. **cuBLAS and cuDNN Workspaces:** takes around 150 to 200 MB of pre-allocated scratchpad memory that NVIDIA libraries grab for GEMM operations.
3. **PyTorch Caching Allocator:** takes another 100 to 150 MB in pre-allocated memory pool blocks to prevent slow `cudaMalloc` operating system syscalls during forward passes.

On my 4 GB laptop GPU, losing 450 MB to overheads plus 716 MB to tied weights means over a quarter of the card is gone before taking in any prompt.

#### KV Cache Growth Per Token

Next comes the KV cache. How fast does memory actually grow as generation proceeds?

Let us calculate the exact bytes by hand:

2 tensors for Key and Value * 2 bytes in BF16 * 2 KV heads * 64 head dimension * 24 layers = **12.288 KB per token**

<p>

![KV Cache Memory Footprint Growth Per Token Calculation for Qwen 2.5 GQA 2 KV Heads](images/qwen2-5-kv-cache-memory-growth-per-token.png)
*KV Cache Memory Footprint Growth Per Token Calculation for Qwen 2.5 GQA 2 KV Heads*

</p>
At a full 32,768 context length, the total KV cache footprint is:

12.288 KB * 32,768 = **402.65 MB**

This is why Grouped Query Attention is so massive. Because Qwen 2.5 0.5B uses only 2 KV heads instead of 14, the KV cache stays around 400 MB instead of ballooning to 2.8 GB.

On the roofline model, token throughput during decode is strictly bound by:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Roofline tok/s</mtext><mo>=</mo><mfrac><mtext>GPU Memory Bandwidth</mtext><mtext>Active Model Bytes Loaded Per Step</mtext></mfrac></mrow><annotation encoding="application/x-tex">\text{Roofline tok/s} = \frac{\text{GPU Memory Bandwidth}}{\text{Active Model Bytes Loaded Per Step}}</annotation></semantics></math>Roofline tok/s=Active Model Bytes Loaded Per StepGPU Memory Bandwidth​
When your KV cache is small, generation speed is limited entirely by how fast you can stream the model weights from GPU memory into the compute cores.

If you want to master the math behind KV cache sizing, arithmetic intensity, and memory bandwidth limits, Kipply’s [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) is the holy grail for calculating exact hardware rooflines.

### Building the Bare-Metal Model Graph

When running inference in Hugging Face through `Qwen2ForCausalLM`, every forward pass carries framework overhead:

- Input checks, device validation, and tensor reshaping on every call.

- Dynamic past_key_values management that appends to cache tensors with repeated heap allocations.

- Separate Python function calls and hooks per layer.

For an inference engine where we want to benchmark and swap CUDA kernels cleanly, we need direct control over the model graph.

In `src/model.py`, I wrote a lightweight implementation of the model graph using PyTorch `nn.Module`. It holds the weights, sets up the 24 layers, and directly calls our custom CUDA extensions for RMSNorm, RoPE, decode attention, GEMM, and SwiGLU.

### How Each Token Moves Through the Pipeline

Here is how each token moves through the pipeline:

<p>

![QUIC Transformer Inference Pipeline Diagram: Embedding, Attention, SwiGLU MLP, and Device-Level Argmax Sampler](images/quic-transformer-inference-pipeline-architecture.png)
*QUIC Transformer Inference Pipeline Diagram: Embedding, Attention, SwiGLU MLP, and Device-Level Argmax Sampler*

</p>
Following the diagram, incoming tokens first pass through the embedding layer and then flow sequentially through all 24 decoder layers.

Inside each layer, the compute divides into two main sections:

In the attention section, the hidden state with shape B * S * 896 passes through our vectorized input RMSNorm and feeds into cuBLASLt GEMM to produce queries, keys, and values. Because of GQA, query has 14 heads while key and value only have 2 heads. We apply RoPE to rotate Q and K in place, store the new K and V in the KV cache, and run attention. During prompt prefill this uses PyTorch SDPA, while during token generation it routes to our custom Split-K GQA kernel. The resulting contextualized tensor is projected back to dimension 896 through an output GEMM and added to the original residual input.

In the MLP section, the normalized tensor enters the feed-forward network through two parallel GEMMs, gate and up projections, that expand the dimension from 896 to 4864. Our fused SwiGLU kernel computes SiLU on the gate and multiplies it by up in a single pass. A down projection GEMM then compresses the 4864 intermediate dimension back to 896, followed by the second residual addition.

Once the tensor completes all 24 layers, it passes through a final RMSNorm, the tied lm_head GEMM to produce vocabulary logits, and our on-GPU argmax sampler to emit the next token ID directly on the device.

### Tokenizer: Keeping It Lightweight

The tokenizer does not have a big impact on GPU inference performance, so there is no need to overcomplicate it.

I just wanted to keep the codebase clean and avoid dragging in the entire transformers library with all its wrapper classes. In `src/tokenizer.py`, I used Hugging Face’s lightweight Rust `tokenizers` library directly. It loads `tokenizer.json`, handles encoding, and applies chat templates cleanly in Rust without extra framework baggage.

<hr>

---

## Writing the Core CUDA Workhorses

Once the bare-metal graph was set up, it was time to tackle the core operators.

When people benchmark LLMs, they usually talk about GEMM and Attention. But during token generation, when batch size is small and sequence length is 1 token, the GPU spends a lot of time bouncing through memory bound operations.

If you leave RMSNorm, RoPE, and SwiGLU as stock PyTorch operations, you pay a heavy price in kernel launch overhead and unnecessary memory traffic. Here is how I implemented each one in CUDA, what broke in eager mode, and the exact speedups measured on an NVIDIA H100.

### RMSNorm: Vectorized 16-Byte Loads and Warp Shuffle

Layer normalization in modern transformers does not subtract the mean. It uses Root Mean Square normalization. For a hidden state vector x with dimension d = 896:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>RMSNorm</mtext><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mi>x</mi><msqrt><mrow><mfrac><mn>1</mn><mi>d</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>d</mi></munderover><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup><mo>+</mo><mi>ϵ</mi></mrow></msqrt></mfrac><mo>⊙</mo><mi>γ</mi></mrow><annotation encoding="application/x-tex">\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma</annotation></semantics></math>RMSNorm(x)=d1​∑i=1d​xi2​+ϵ<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.88em" viewBox="0 0 400000 1944" preserveAspectRatio="xMinYMin slice"><path d="M983 90
l0 -0
c4,-6.7,10,-10,18,-10 H400000v40
H1013.1s-83.4,268,-264.1,840c-180.7,572,-277,876.3,-289,913c-4.7,4.7,-12.7,7,-24,7
s-12,0,-12,0c-1.3,-3.3,-3.7,-11.7,-7,-25c-35.3,-125.3,-106.7,-373.3,-214,-744
c-10,12,-21,25,-33,39s-32,39,-32,39c-6,-5.3,-15,-14,-27,-26s25,-30,25,-30
c26.7,-32.7,52,-63,76,-91s52,-60,52,-60s208,722,208,722
c56,-175.3,126.3,-397.3,211,-666c84.7,-268.7,153.8,-488.2,207.5,-658.5
c53.7,-170.3,84.5,-266.8,92.5,-289.5z
M1001 80h400000v40h-400000z"></path></svg>​x​⊙γ
Each element is divided by the root mean square of the entire row and scaled by a learnable parameter vector gamma.

<p>

![RMSNorm Kernel Optimization: Vectorized 16-Byte Loads and In-Register Warp Shuffle Reduction](images/rmsnorm-kernel-fusion-vectorized-warp-shuffle.png)
*RMSNorm Kernel Optimization: Vectorized 16-Byte Loads and In-Register Warp Shuffle Reduction*

</p>
In Hugging Face and stock PyTorch, RMSNorm runs like this:

```
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

This single line of Python issues four separate CUDA kernel launches:

1. The squaring operation squares every element and writes an intermediate tensor to GPU DRAM.
2. The mean reduction reads that tensor back from DRAM, sums across the hidden dimension, divides by 896, and writes another tensor to DRAM.
3. The reciprocal square root with epsilon reads the mean, adds epsilon, computes the rsqrt, and writes back to DRAM.
4. The final multiplication reads the original input x, reads the scale factor, reads the weight vector, multiplies them, and writes the output to DRAM.

For small batch sizes or single token generation, each kernel launch incurs around 3 to 5 microseconds of driver dispatch overhead. Worse, you are round tripping megabytes of transient memory through global DRAM when the entire 896 dimension fits comfortably inside the processor registers.

In `kernels/cuda/rmsnorm.cu`, I wrote a single fused kernel:

- Exactly 128 threads in one thread block process an entire row of 896 FP16 values.

- To maximize memory bus efficiency, I used 16-byte aligned vectorized reads. I defined a `Half8` struct packed with four `half2` pairs:

```
struct alignas(16) Half8 {
    half2 h0, h1, h2, h3;
};
```

Since 896 divided by 8 is 112, 112 threads each issue a single 16-byte instruction to load 8 values straight into registers and shared memory. Each thread squares its 8 values locally, and then we run a fast warp reduction using `__shfl_down_sync`:

```
__device__ float warpReduce(float val) {
    #pragma unroll
    for (int activeThreads = WARP_SIZE >> 1; activeThreads > 0; activeThreads >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, activeThreads);
    }
    return val;
}
```

In 5 warp shuffle steps, from 16 down to 1, the threads exchange values in registers without touching shared memory. Lane 0 computes reciprocal square root in shared memory, and each thread scales its 8 values and writes them back in a single pass. Zero transient DRAM allocations.

Measured head to head on an NVIDIA H100 80GB GPU against Hugging Face eager execution:

<table><thead><tr><th align="center">Tokens M</th><th align="center">Custom Kernel Latency</th><th align="center">HF Eager Latency</th><th align="center">Speedup</th><th align="center">Max Absolute Error</th></tr></thead><tbody><tr><td align="center">**1**</td><td align="center">**13.28 us**</td><td align="center">68.45 us</td><td align="center">**5.15x**</td><td align="center">0.0</td></tr><tr><td align="center">**4**</td><td align="center">**12.99 us**</td><td align="center">67.33 us</td><td align="center">**5.18x**</td><td align="center">0.0</td></tr><tr><td align="center">**16**</td><td align="center">**13.12 us**</td><td align="center">68.35 us</td><td align="center">**5.21x**</td><td align="center">6.10e-5</td></tr><tr><td align="center">**64**</td><td align="center">**13.22 us**</td><td align="center">68.67 us</td><td align="center">**5.20x**</td><td align="center">4.88e-4</td></tr><tr><td align="center">**256**</td><td align="center">**13.28 us**</td><td align="center">69.57 us</td><td align="center">**5.24x**</td><td align="center">9.77e-4</td></tr><tr><td align="center">**1024**</td><td align="center">**13.34 us**</td><td align="center">70.08 us</td><td align="center">**5.25x**</td><td align="center">1.95e-3</td></tr><tr><td align="center">**4096**</td><td align="center">**13.63 us**</td><td align="center">71.33 us</td><td align="center">**5.23x**</td><td align="center">1.95e-3</td></tr></tbody></table>
Latency dropped from around 70 microseconds down to **13 microseconds flat**, delivering a solid **5.2x speedup** across every batch size with zero precision loss.

### RoPE: Keeping Trig and Rotations in Registers

Rotary Position Embedding encodes token positions by rotating pairs of coordinates in the complex plane. For each attention head with dimension D = 64, elements are paired into 32 two-dimensional coordinates:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mrow><mo fence="true">[</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>o</mi><mi>u</mi><msub><mi>t</mi><mn>0</mn></msub></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>o</mi><mi>u</mi><msub><mi>t</mi><mn>1</mn></msub></mrow></mstyle></mtd></mtr></mtable><mo fence="true">]</mo></mrow><mo>=</mo><mrow><mo fence="true">[</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>cos</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mi>sin</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>sin</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>cos</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mstyle></mtd></mtr></mtable><mo fence="true">]</mo></mrow><mrow><mo fence="true">[</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>v</mi><mn>0</mn></msub></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>v</mi><mn>1</mn></msub></mstyle></mtd></mtr></mtable><mo fence="true">]</mo></mrow></mrow><annotation encoding="application/x-tex">\begin{bmatrix} out_0 \\ out_1 \end{bmatrix} = \begin{bmatrix} \cos(m \theta_i) &#x26; -\sin(m \theta_i) \\ \sin(m \theta_i) &#x26; \cos(m \theta_i) \end{bmatrix} \begin{bmatrix} v_0 \\ v_1 \end{bmatrix}</annotation></semantics></math>[out0​out1​​]=[cos(mθi​)sin(mθi​)​−sin(mθi​)cos(mθi​)​][v0​v1​​]
Expanding the matrix multiplication gives:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>o</mi><mi>u</mi><msub><mi>t</mi><mn>0</mn></msub><mo>=</mo><msub><mi>v</mi><mn>0</mn></msub><mi>cos</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mo>−</mo><msub><mi>v</mi><mn>1</mn></msub><mi>sin</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">out_0 = v_0 \cos(m \theta_i) - v_1 \sin(m \theta_i)</annotation></semantics></math>out0​=v0​cos(mθi​)−v1​sin(mθi​)
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>o</mi><mi>u</mi><msub><mi>t</mi><mn>1</mn></msub><mo>=</mo><msub><mi>v</mi><mn>0</mn></msub><mi>sin</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mo>+</mo><msub><mi>v</mi><mn>1</mn></msub><mi>cos</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><msub><mi>θ</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">out_1 = v_0 \sin(m \theta_i) + v_1 \cos(m \theta_i)</annotation></semantics></math>out1​=v0​sin(mθi​)+v1​cos(mθi​)
Here m is the sequence position, and frequency is defined as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>θ</mi><mi>i</mi></msub><mo>=</mo><msup><mn>1000000</mn><mrow><mo>−</mo><mn>2</mn><mi>i</mi><mi mathvariant="normal">/</mi><mi>D</mi></mrow></msup></mrow><annotation encoding="application/x-tex">\theta_i = 1000000^{-2i/D}</annotation></semantics></math>θi​=1000000−2i/D
The standard implementation in Hugging Face calls `rotate_half`:

```
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Inside forward:
(q * cos) + (rotate_half(q) * sin)
```

Look at what this does under the hood:

1. It slices the tensor into two halves, allocating new tensor memory.
2. It negates one half and concatenates them back together, forcing a full memory copy of the entire attention tensor.
3. It broadcasts cosine and sine tables across batch and head dimensions.
4. It executes two separate elementwise multiplications followed by an elementwise addition.

This causes continuous tensor slicing, memory thrashing, and repeated kernel dispatch for every single layer in the model.

In `kernels/cuda/rope.cu`:

- All trigonometric frequencies are precomputed once at engine startup via `qwen2_rope_table_init` and stored as packed FP16 tables directly in GPU memory.

- Exactly 1 warp of 32 threads processes one entire attention head of 64 dimensions.

- Thread `lane_id` reads v0 from the first half and v1 from the second half directly into registers:

```
float v0 = __half2float(src[idx0]);
float v1 = __half2float(src[idx1]);

float out0 = v0 * cos_val - v1 * sin_val;
float out1 = v0 * sin_val + v1 * cos_val;
dst[idx0] = __float2half(out0);
dst[idx1] = __float2half(out1);
```

The rotation happens strictly in registers. The output is written back to global memory in a single pass. I also added batch stride indexing so left-padded batches can look up their exact position offsets directly on GPU without allocating dynamic position masks on the host.

Measured on an NVIDIA H100 with 14 heads and head dimension 64:

<table><thead><tr><th align="center">Sequence Length S</th><th align="center">Custom Kernel Latency</th><th align="center">HF Eager Latency</th><th align="center">Speedup</th><th align="center">Max Absolute Error</th></tr></thead><tbody><tr><td align="center">**1 Decode**</td><td align="center">**12.96 us**</td><td align="center">71.17 us</td><td align="center">**5.49x**</td><td align="center">0.0</td></tr><tr><td align="center">**32**</td><td align="center">**13.18 us**</td><td align="center">71.39 us</td><td align="center">**5.42x**</td><td align="center">3.91e-3</td></tr><tr><td align="center">**128**</td><td align="center">**13.02 us**</td><td align="center">72.96 us</td><td align="center">**5.60x**</td><td align="center">3.91e-3</td></tr><tr><td align="center">**512**</td><td align="center">**13.76 us**</td><td align="center">71.65 us</td><td align="center">**5.21x**</td><td align="center">3.91e-3</td></tr><tr><td align="center">**1024**</td><td align="center">**17.82 us**</td><td align="center">82.21 us</td><td align="center">**4.61x**</td><td align="center">3.91e-3</td></tr><tr><td align="center">**2048**</td><td align="center">**27.71 us**</td><td align="center">125.25 us</td><td align="center">**4.52x**</td><td align="center">3.91e-3</td></tr></tbody></table>
During single-token decode, latency crashed down from **71.17 us to 12.96 us**, giving an immediate **5.49x speedup** by keeping tensor slicing and trigonometry off the memory bus.

### SwiGLU: Fusing Activation and Multiplication

The feed-forward network in Qwen 2.5 expands hidden dimension 896 to intermediate dimension 4864 using SwiGLU. Given gate projection g and up projection u:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>SwiGLU</mtext><mo stretchy="false">(</mo><mi>g</mi><mo separator="true">,</mo><mi>u</mi><mo stretchy="false">)</mo><mo>=</mo><mtext>SiLU</mtext><mo stretchy="false">(</mo><mi>g</mi><mo stretchy="false">)</mo><mo>⊙</mo><mi>u</mi><mo>=</mo><mrow><mo fence="true">[</mo><mfrac><mi>g</mi><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>g</mi></mrow></msup></mrow></mfrac><mo fence="true">]</mo></mrow><mo>⊙</mo><mi>u</mi></mrow><annotation encoding="application/x-tex">\text{SwiGLU}(g, u) = \text{SiLU}(g) \odot u = \left[ \frac{g}{1 + e^{-g}} \right] \odot u</annotation></semantics></math>SwiGLU(g,u)=SiLU(g)⊙u=[1+e−gg​]⊙u
<p>

![SwiGLU Fused Kernel Memory Optimization: Eliminating Intermediate SiLU DRAM Traffic](images/swiglu-kernel-fusion-memory-bandwidth-optimization.png)
*SwiGLU Fused Kernel Memory Optimization: Eliminating Intermediate SiLU DRAM Traffic*

</p>
In eager PyTorch, this is written as:

```
F.silu(gate) * up
```

Because these are separate PyTorch operations, the runtime executes two distinct GPU kernels:

- Kernel 1 reads `gate` from DRAM, computes the SiLU activation curve, and writes that intermediate tensor back to DRAM.

- Kernel 2 reads that intermediate tensor back from DRAM, reads `up` from DRAM, multiplies them, and writes the result back to DRAM.

For an intermediate dimension of 4864 across tokens, this requires 3 memory reads and 2 memory writes. In FP16, that is 10 bytes transferred per element. On an activation step that is 100% memory bandwidth bound, reading and writing that transient SiLU tensor wastes 40% of your memory bandwidth on useless traffic.

In `kernels/cuda/swiglu.cu`, I fused the entire operation into a single streaming kernel:

```
const float g = __half2float(gate[idx]);
const float u = __half2float(up[idx]);
y[idx] = __float2half((g / (1.0f + expf(-g))) * u);
```

Each thread loads `gate` and `up` once directly into registers, calculates SiLU using high-speed hardware exponential math, multiplies by `up` in registers, and writes the output once to memory. Global memory traffic drops to exactly 2 reads and 1 write: only 6 bytes per element instead of 10 bytes.

Measured on an NVIDIA H100 with intermediate dimension 4864:

<table><thead><tr><th align="center">Tokens M</th><th align="center">Custom Kernel Latency</th><th align="center">HF Eager Latency</th><th align="center">Speedup</th><th align="center">Max Absolute Error</th></tr></thead><tbody><tr><td align="center">**1**</td><td align="center">**12.83 us**</td><td align="center">19.68 us</td><td align="center">**1.53x**</td><td align="center">3.91e-3</td></tr><tr><td align="center">**4**</td><td align="center">**12.99 us**</td><td align="center">19.84 us</td><td align="center">**1.53x**</td><td align="center">3.91e-3</td></tr><tr><td align="center">**16**</td><td align="center">**12.99 us**</td><td align="center">19.74 us</td><td align="center">**1.52x**</td><td align="center">7.81e-3</td></tr><tr><td align="center">**64**</td><td align="center">**13.12 us**</td><td align="center">19.90 us</td><td align="center">**1.52x**</td><td align="center">7.81e-3</td></tr><tr><td align="center">**256**</td><td align="center">**13.28 us**</td><td align="center">19.71 us</td><td align="center">**1.48x**</td><td align="center">7.81e-3</td></tr><tr><td align="center">**1024**</td><td align="center">**19.62 us**</td><td align="center">22.78 us</td><td align="center">**1.16x**</td><td align="center">7.81e-3</td></tr><tr><td align="center">**4096**</td><td align="center">**70.59 us**</td><td align="center">77.82 us</td><td align="center">**1.10x**</td><td align="center">7.81e-3</td></tr></tbody></table>
During token generation, latency dropped from **19.68 us down to 12.83 us**, delivering a clean **1.53x speedup** while saving 40% of DRAM memory traffic on every single MLP forward pass.

### cuBLASLt GEMM and In-Place Bias Addition

All linear projections, meaning Q, K, V, Out, Gate, Up, and Down, run through `cublasGemmEx`.

If you want to understand what it takes to write bare-metal matrix multiplication from scratch, Aleksa Gordić’s [CUDA Matrix Multiplication deep dive](https://www.aleksagordic.com/blog/matmul) is the guru level guide I studied to understand register tiling, shared memory layouts, and hardware limits before building out these optimizations.

Inputs use FP16 while accumulation runs in FP32 math for numerical stability.

cuBLAS expects column major matrices, while PyTorch stores them row major. By transposing operand order, we pass row major tensors directly to cuBLAS without any memory copies:

```
cublasGemmEx(
    handle(), CUBLAS_OP_T, CUBLAS_OP_N,
    N, M, K,
    &#x26;fal,
    wc.data_ptr&#x3C;at::Half>(), CUDA_R_16F, K,
    x2.data_ptr&#x3C;at::Half>(), CUDA_R_16F, K,
    &#x26;fbe,
    y2.data_ptr&#x3C;at::Half>(), CUDA_R_16F, N,
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

If a projection has bias, an in-place bias addition kernel runs immediately on the same CUDA stream.

### Greedy Argmax Sampler: Staying on the GPU

At the end of all 24 layers, the tied lm_head projection generates logits across the entire vocabulary of 151,936 tokens. The model must pick the single most likely token to emit next. For a vocabulary logit vector z with dimension V = 151,936:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>t</mi><mo>=</mo><msub><mrow><mi mathvariant="normal">argmax</mi><mo>⁡</mo></mrow><mrow><mi>i</mi><mo>∈</mo><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mi>V</mi><mo>−</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></msub><msub><mi>z</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">t = \operatorname{argmax}_{i \in \{0, \dots, V-1\}} z_i</annotation></semantics></math>t=argmaxi∈{0,…,V−1}​zi​
The goal is to find the token index t that maximizes logit score z_i, with deterministic tie breaking on lower indices if two scores match.

In stock Hugging Face and eager pipelines, token selection runs through PyTorch:

```
next_token = torch.argmax(logits, dim=-1)
# Or extracting to Python host:
token_id = next_token.item()
```

This causes two distinct bottlenecks:

1. **Framework dispatch overhead:** `torch.argmax` allocates tensor metadata, dispatches through general ATen reduction kernels, and writes intermediate structures.
2. **Host synchronization stalls:** Pulling logits or calling `.item` forces the CPU to synchronize with the GPU. Copying 151,936 half precision values over the PCIe bus or stalling host execution until the GPU finishes destroys pipeline concurrency.

If the host blocks waiting for logits on every single token, the GPU cannot overlap kernel launches, and decode throughput collapses.

In `kernels/cuda/sampler.cu`, I built an on-device block-level argmax reduction:

- Exactly one thread block of 256 threads handles one batch sequence row.

- The 256 threads execute a grid-stride loop across all 151,936 vocabulary logits in parallel:

```
float best_val = -FLT_MAX;
int64_t best_idx = 0;

for (int64_t i = threadIdx.x; i &#x3C; V; i += kThreads) {
    const float v = __half2float(base[i]);
    if (v > best_val) {
        best_val = v;
        best_idx = i;
    }
}
```

Each thread tracks its own local maximum value and index pair entirely in registers. Threads store their candidate pairs into shared memory and perform a fast shared memory tree reduction across the block:

```
for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x &#x3C; s) {
        const float v = s_val[threadIdx.x + s];
        const int64_t i = s_idx[threadIdx.x + s];
        if (v > s_val[threadIdx.x] || (v == s_val[threadIdx.x] &#x26;&#x26; i &#x3C; s_idx[threadIdx.x])) {
            s_val[threadIdx.x] = v;
            s_idx[threadIdx.x] = i;
        }
    }
    __syncthreads();
}
```

Thread 0 writes the winning 64-bit integer token ID directly into global memory:

```
if (threadIdx.x == 0) {
    out[row] = s_idx[0];
}
```

The CPU never touches the 151,936 logit floats. The entire logits matrix stays inside GPU memory, and the CPU only inspects the final integer token ID when decoding finishes.

Measured head to head on an NVIDIA H100 with vocabulary size 151,936 against Hugging Face eager argmax:

<table><thead><tr><th align="center">Batch Size B</th><th align="center">Custom Kernel Latency</th><th align="center">HF Eager Latency</th><th align="center">Speedup</th><th align="center">Correctness</th></tr></thead><tbody><tr><td align="center">**1**</td><td align="center">26.21 us</td><td align="center">20.54 us</td><td align="center">0.78x</td><td align="center">Exact Match 100%</td></tr><tr><td align="center">**4**</td><td align="center">26.46 us</td><td align="center">20.35 us</td><td align="center">0.77x</td><td align="center">Exact Match 100%</td></tr><tr><td align="center">**16**</td><td align="center">26.62 us</td><td align="center">22.75 us</td><td align="center">0.85x</td><td align="center">Exact Match 100%</td></tr><tr><td align="center">**32**</td><td align="center">**26.88 us**</td><td align="center">26.85 us</td><td align="center">**1.00x**</td><td align="center">Exact Match 100%</td></tr><tr><td align="center">**64**</td><td align="center">**26.91 us**</td><td align="center">31.90 us</td><td align="center">**1.19x**</td><td align="center">Exact Match 100%</td></tr></tbody></table>
While raw standalone kernel latency is on par with ATen, the real architectural win is system level:

- It eliminates the PCIe transfer and host synchronization bubble entirely.

- It keeps the generation loop completely non-blocking on the device from token 1 to token 100.

- Correctness is 100% exact match against eager PyTorch across every batch size.

<hr>

---

## The Attention Frontier: Prefill vs Decode

Attention is where inference engines either fly or fall apart.

When people say they use FlashAttention, they often treat it as a single magic function that solves everything. But in an autoregressive model, prompt prefill and token decode are two entirely different worlds running on completely different hardware constraints. For the theoretical arithmetic on why prefill is compute bound while decode is memory bandwidth bound, Kipply’s [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) lays out the exact formulas.

Here is the deep dive into how attention actually behaves under the hood, how eager PyTorch blows up memory, and how we solved both prefill and decode.

### Qwen 2.5 GQA: 14 Query Heads to 2 KV Heads

Before touching the implementation, look at how Qwen 2.5 routes attention:

<p>

![Grouped Query Attention GQA Diagram: 14 Query Heads Sharing 2 Key-Value Heads in Qwen 2.5](images/qwen2-5-grouped-query-attention-gqa-ratio.png)
*Grouped Query Attention GQA Diagram: 14 Query Heads Sharing 2 Key-Value Heads in Qwen 2.5*

</p>
Instead of standard multi-head attention where every query head has its own key and value head, Qwen 2.5 0.5B groups 14 query heads across only 2 key-value heads. That gives a 7 to 1 GQA ratio.

Each head has dimension 64:

- Query projection produces 14 heads * 64 = 896 channels.

- Key and Value projections produce 2 heads * 64 = 128 channels each.

In the forward pass, each group of 7 query heads shares the exact same key and value tensors. After computing softmax of Q times K transpose divided by the square root of head dimension and multiplying by V, the 14 head outputs are concatenated and projected through Wo back to hidden dimension 896.

This grouping dramatically reduces KV cache footprint, but it creates a massive challenge for decode parallelization.

### The Eager Attention Memory Trap and the 24.45K Crash Limit

When I was testing long prompts on an 80 GB NVIDIA H100, I wanted to see how far Hugging Face eager attention could go before running out of memory.

On paper, an 80 GB H100 has roughly 78.2 GB of free usable space after taking out the 0.95 GB driver and runtime overheads and the 0.92 GB model weights. What decides where VRAM reaches its maximum and the program goes out of memory is the biggest VRAM space used by a function at the same time:

<p>

![Peak VRAM Allocation by Operator Function in PyTorch Transformer Forward Pass](images/peak-vram-usage-by-operator-function.png)
*Peak VRAM Allocation by Operator Function in PyTorch Transformer Forward Pass*

</p>
You would think you could easily run 30,000 or 40,000 tokens of context on a tiny 0.5B model. Instead, standard eager execution crashed hard at exactly **24.45K tokens**!

Why? Look at how standard eager attention executes:

<p>

![Eager Attention Quadratic Memory Explosion and the 24.45K Context Crash Limit](images/eager-attention-quadratic-vram-memory-explosion.png)
*Eager Attention Quadratic Memory Explosion and the 24.45K Context Crash Limit*

</p>
In eager PyTorch, attention is not fused. It allocates separate intermediate tensor grids in GPU DRAM:

1. Raw Scores in FP16: 1 * M
2. Scaled Scores in FP16: 1 * M
3. Softmax Input converted to FP32: 2 * M
4. Softmax Output converted to FP32: 2 * M

That equals **5 * M bytes of transient memory per attention head**!

When you sum this across 24 layers, the total peak memory spike approximates to:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Peak Spike</mtext><mo>≈</mo><mtext>layers</mtext><mo>×</mo><mn>5</mn><mo>×</mo><msup><mtext>ctx</mtext><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">\text{Peak Spike} \approx \text{layers} \times 5 \times \text{ctx}^2</annotation></semantics></math>Peak Spike≈layers×5×ctx2
Because context is squared, memory does not grow linearly. At context 16,384, the intermediate attention spike alone eats **35.84 GB** of VRAM! When pushing context toward 25,000 tokens, that quadratic spike hits 79.37 GB, completely swallowing the 78.2 GB available headroom and crashing the GPU with out of memory.

### The 2,170-Token Crossover: When LM Head Dominates

Here is another subtle insight from profiling. The peak memory consumed during a forward pass is determined by the competition between two operations:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Peak VRAM</mtext><mo>=</mo><mi>max</mi><mo>⁡</mo><mo stretchy="false">{</mo><mtext>Peak Attention Spike</mtext><mo separator="true">,</mo><mtext>  </mtext><mtext>Peak LM Head Logits Projection</mtext><mo stretchy="false">}</mo></mrow><annotation encoding="application/x-tex">\text{Peak VRAM} = \max\{ \text{Peak Attention Spike},\; \text{Peak LM Head Logits Projection} \}</annotation></semantics></math>Peak VRAM=max{Peak Attention Spike,Peak LM Head Logits Projection}
At the end of the model, `lm_head` projects hidden states of shape `[S, 896]` to vocabulary logits of shape `[S, 151936]`. In FP16, that buffer requires:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>S</mi><mo>×</mo><mn>151</mn><mo separator="true">,</mo><mn>936</mn><mo>×</mo><mn>2</mn><mtext> bytes</mtext></mrow><annotation encoding="application/x-tex">S \times 151,936 \times 2 \text{ bytes}</annotation></semantics></math>S×151,936×2 bytes
Now set the attention spike equal to the LM head projection:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mn>2</mn><mo>×</mo><mn>14</mn><mo>×</mo><mn>5</mn><mo>×</mo><msup><mi>S</mi><mn>2</mn></msup><mo>=</mo><mn>151</mn><mo separator="true">,</mo><mn>936</mn><mo>×</mo><mi>S</mi><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">2 \times 14 \times 5 \times S^2 = 151,936 \times S \times 2</annotation></semantics></math>2×14×5×S2=151,936×S×2
Solving for sequence length S gives:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>S</mi><mo>≈</mo><mn>2170</mn><mtext> tokens</mtext></mrow><annotation encoding="application/x-tex">S \approx 2170 \text{ tokens}</annotation></semantics></math>S≈2170 tokens
<p>

![The 2170-Token Crossover Point: LM Head Logits Projection vs Quadratic Attention VRAM Allocation](images/lm-head-projection-vs-attention-vram-crossover.png)
*The 2170-Token Crossover Point: LM Head Logits Projection vs Quadratic Attention VRAM Allocation*

</p>
This 2170 token threshold is critical:

- **Below 2170 tokens:** The LM head vocabulary projection sets the peak memory of the model. At context 512, LM head needs 148 MB while attention only needs 35 MB. Replacing eager attention with FlashAttention saves exactly **0 MB** of peak VRAM!

- **Above 2170 tokens:** The quadratic attention curve takes off and completely overwhelms the LM head. At 8192 context, FlashAttention saves 6.13 GB. At 16,384 context, it saves 28.9 GB. At 32,768 context, eager execution crashes into out of memory while FlashAttention cruises comfortably at only 10.9 GB!

To fix that quadratic explosion during prefill, you have two choices:

<p>

![Comparison of Memory-Efficient Tiled Attention vs Online Softmax FlashAttention](images/memory-efficient-vs-flash-attention-tiling.png)
*Comparison of Memory-Efficient Tiled Attention vs Online Softmax FlashAttention*

</p>
1. **Memory-Efficient Attention:** Tiles the sequence matrix into small blocks like 64 by 64. Because each tile is small, the intermediate tensor M shrinks. This eliminates the memory spike, but still moves data between shared memory and global memory.
2. **FlashAttention:** Combines tiling with mathematical fusion using online softmax. Intermediate attention matrices never get written to GPU DRAM. They stay entirely inside registers and shared memory. This cuts memory footprint and dramatically reduces memory bandwidth traffic.

### Prefill in Practice: TMA, WGMMA, and an Honest Choice

For prompt prefill on Hopper, I wanted to push the hardware to its absolute limit using specialized sm_90a instructions:

- **TMA:** Asynchronous copying of multidimensional tensor tiles directly from global memory into shared memory via `cuTensorMapEncodeTiled`, bypassing register files completely.

- **WGMMA:** Asynchronous 64-thread tensor core math executing directly on shared memory operands through `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16`.

- **128B Swizzling:** Conflict-free shared memory layout via `make_smem_desc`.

- **Double Buffering:** A 2-stage ping-pong pipeline overlapping global memory data movement with tensor core compute.

In `kernels/cuda/flash_attn_prefill.cu`, I built this hand-written Hopper kernel. At B=1, S=128, it clocked in at **32.04 us** reaching 0.92 TFLOPs, beating FlashAttention-2 at 45.61 us on raw standalone compute.

However, when plugging it into a real inference serving stack with arbitrary prompt lengths, left-hand padding, and dynamic batching, reality hit:

- Handling arbitrary variable sequence lengths and causal masks required complex padding tracking inside the TMA descriptors.

- In edge cases, standalone custom prefill ended up performing worse or becoming unstable compared to existing tuned implementations.

- Meanwhile, PyTorch native `F.scaled_dot_product_attention` on Hopper already calls NVIDIA tuned FlashAttention-3 and cuDNN C++ backend under the hood with zero Python overhead.

So I created `kernels/flash_attn_prefill_pytorch.py` to wrap PyTorch SDPA directly for the prefill stage. It handles arbitrary sequence lengths, causal masking, and left-padding cleanly without stability risks.

Decode attention, however, was an entirely different story. That is where standard implementations fall apart.

### The Decode Bottleneck: 132 SMs Starving for Work

Prefill is compute bound. Decode is memory.

During token generation, you process only one new token at a time: sequence length S = 1. The Query is a single token vector `[1, 14, 64]`. Key and Value are long history matrices `[S_kv, 2, 64]` stored in the KV cache. This is a vector-matrix multiplication, strictly limited by GPU memory bandwidth.

Now look at GPU occupancy:

- Qwen 2.5 0.5B has 14 query heads.

- If you launch standard FlashAttention at Batch Size 1, you launch exactly **14 thread blocks**, one per query head.

- An NVIDIA H100 GPU has **132 Streaming Multiprocessors**.

With only 14 thread blocks in flight, **118 SMs sit completely idle**! Over 85% of your GPU silicon is doing nothing while waiting for the KV cache to stream from memory. This is why small models get pitiful decode throughput on massive GPUs unless you parallelize across the cache itself.

### Split-K GQA Decode Attention: Lighting Up the GPU

To wake up all 132 SMs during decode, we use **Split-K Attention** in `kernels/cuda/flash_attn_decode.cu`.

Standard attention computes softmax across the full sequence length S:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Attention</mtext><mo stretchy="false">(</mo><mi>Q</mi><mo separator="true">,</mo><mi>K</mi><mo separator="true">,</mo><mi>V</mi><mo stretchy="false">)</mo><mo>=</mo><mtext>softmax</mtext><mrow><mo fence="true">(</mo><mfrac><mrow><mi>Q</mi><msup><mi>K</mi><mi>T</mi></msup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo fence="true">)</mo></mrow><mi>V</mi></mrow><annotation encoding="application/x-tex">\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V</annotation></semantics></math>Attention(Q,K,V)=softmax(dk​<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
c69,-144,104.5,-217.7,106.5,-221
l0 -0
c5.3,-9.3,12,-14,20,-14
H400000v40H845.2724
s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
M834 80h400000v40h-400000z"></path></svg>​QKT​)V
In Split-K, we partition the key-value sequence length S into K independent chunks of 64 tokens. Each split k computes its own local output Ok, local maximum mk, and local normalizer lk using online softmax. A fast combine pass then merges the splits using the rescaling identity:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>α</mi><mi>k</mi></msub><mo>=</mo><mi>exp</mi><mo>⁡</mo><mo stretchy="false">(</mo><msub><mi>m</mi><mi>k</mi></msub><mo>−</mo><msub><mi>m</mi><mtext>global</mtext></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\alpha_k = \exp(m_k - m_{\text{global}})</annotation></semantics></math>αk​=exp(mk​−mglobal​)
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>O</mi><mtext>final</mtext></msub><mo>=</mo><mfrac><mrow><munder><mo>∑</mo><mi>k</mi></munder><msub><mi>α</mi><mi>k</mi></msub><msub><mi>O</mi><mi>k</mi></msub></mrow><mrow><munder><mo>∑</mo><mi>k</mi></munder><msub><mi>α</mi><mi>k</mi></msub><msub><mi>l</mi><mi>k</mi></msub></mrow></mfrac></mrow><annotation encoding="application/x-tex">O_{\text{final}} = \frac{\sum_k \alpha_k O_k}{\sum_k \alpha_k l_k}</annotation></semantics></math>Ofinal​=∑k​αk​lk​∑k​αk​Ok​​
Instead of assigning one whole sequence to one thread block, we slice the historical KV sequence length into chunks of 64 tokens:

1. If context length is 1024 tokens, dividing by 64 produces 16 splits.
2. We launch a 3D CUDA grid:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Grid</mtext><mo>=</mo><mo stretchy="false">[</mo><mtext>num_splits</mtext><mo separator="true">,</mo><mtext>  </mtext><msub><mi>n</mi><mrow><mi>k</mi><mi>v</mi></mrow></msub><mo separator="true">,</mo><mtext>  </mtext><mi>B</mi><mo stretchy="false">]</mo></mrow><annotation encoding="application/x-tex">\text{Grid} = [\text{num\_splits},\; n_{kv},\; B]</annotation></semantics></math>Grid=[num_splits,nkv​,B]
For Batch Size 1 with 16 splits and 2 KV heads, we launch 32 blocks instead of 14. For Batch Size 4, that becomes 128 blocks, instantly lighting up almost every SM on the H100!

Since 7 query heads share 1 KV head:

- Each thread block loads its 64-token KV chunk into shared memory once.

- 7 warps inside the block compute attention for all 7 query heads in parallel directly out of shared memory.

<li>Each warp maintains local partial sums in registers using `__shfl_xor_sync` online softmax:
<ul>
- Partial output: `O_part`

- Partial maximum: `M_part`

- Partial normalizer sum: `L_part`

</li>
</ul>
A lightweight second-pass combine kernel `fad_combine_kernel` then merges the splits across the sequence dimension.

### The Left-Padding Clamping Bug and the NaN Fix

While running batched generation benchmarks, I ran into a subtle bug where the model suddenly spat out NaN values, corrupting the generation.

In batched serving, prompts have variable lengths. Shorter prompts are padded on the left with zero tokens:

```
Sequence 0: [Pad, Pad, Pad, Pad, Token1, Token2, Token3]
Sequence 1: [TokenA, TokenB, TokenC, TokenD, TokenE, ...]
```

In standard attention, padded tokens are masked by setting their scores to negative infinity. In Split-K decode, what happens when an entire 64-token split falls inside the left padding?

- Every token in that chunk is masked to negative infinity.

- The split maximum is negative infinity.

- When computing the online softmax rescale factor, subtracting negative infinity from negative infinity produces NaN!

That single NaN poisoned the accumulator and killed the sequence.

Instead of letting threads compute math on dummy padding tokens, I clamped the split range directly against the sequence padding offset:

```
const int pad = pad_lens ? pad_lens[b] : 0;
const int token_start = max(split * chunk, pad);
const int token_end = min(split * chunk + chunk, S);

if (token_start >= token_end) return;
```

If an entire 64-token chunk falls inside padding, the thread block exits immediately without touching floating point math. Its partial buffers stay initialized to clean sentinel values. In the second-pass combine kernel, if a split has negative infinity maximum, it simply outputs zero. Zero wasted cycles, and zero NaNs.

### Attention Benchmarks: Standalone Win to End-to-End Payoff

Here are the standalone microbenchmark numbers measured directly on an NVIDIA H100 GPU against PyTorch native SDPA:

<table><thead><tr><th align="center">Batch Size B</th><th align="center">KV Length S</th><th align="center">Custom Decode Latency</th><th align="center">PyTorch SDPA Latency</th><th align="center">Speedup</th><th align="center">Effective Memory Bandwidth</th></tr></thead><tbody><tr><td align="center">**1**</td><td align="center">128</td><td align="center">**29.59 us**</td><td align="center">42.13 us</td><td align="center">**1.42x**</td><td align="center">2.34 GB/s</td></tr><tr><td align="center">**1**</td><td align="center">512</td><td align="center">**30.40 us**</td><td align="center">44.32 us</td><td align="center">**1.46x**</td><td align="center">8.74 GB/s</td></tr><tr><td align="center">**1**</td><td align="center">1024</td><td align="center">**32.68 us**</td><td align="center">44.37 us</td><td align="center">**1.36x**</td><td align="center">16.16 GB/s</td></tr><tr><td align="center">**1**</td><td align="center">2048</td><td align="center">**36.38 us**</td><td align="center">46.07 us</td><td align="center">**1.27x**</td><td align="center">28.92 GB/s</td></tr><tr><td align="center">**4**</td><td align="center">1024</td><td align="center">**33.72 us**</td><td align="center">51.50 us</td><td align="center">**1.53x**</td><td align="center">62.62 GB/s</td></tr><tr><td align="center">**4**</td><td align="center">4096</td><td align="center">**53.22 us**</td><td align="center">114.13 us</td><td align="center">**2.14x**</td><td align="center">157.88 GB/s</td></tr><tr><td align="center">**16**</td><td align="center">1024</td><td align="center">**39.36 us**</td><td align="center">112.32 us</td><td align="center">**2.85x**</td><td align="center">214.58 GB/s</td></tr><tr><td align="center">**16**</td><td align="center">4096</td><td align="center">**81.14 us**</td><td align="center">370.80 us</td><td align="center">**4.57x**</td><td align="center">414.22 GB/s</td></tr><tr><td align="center">**32**</td><td align="center">2048</td><td align="center">**72.74 us**</td><td align="center">390.75 us</td><td align="center">**5.37x**</td><td align="center">462.86 GB/s</td></tr><tr><td align="center">**32**</td><td align="center">4096</td><td align="center">**133.24 us**</td><td align="center">720.83 us</td><td align="center">**5.41x**</td><td align="center">504.55 GB/s</td></tr></tbody></table>
Look at how the speedup scales as batch size and context grow:

- At small batches, Split-K eliminates SM starvation, boosting latency by 1.36x to 1.53x.

- At Batch 32 and Context 4096, decode attention latency drops from 720.83 us in PyTorch SDPA down to **133.24 us**, an incredible **5.41x speedup** while sustaining over 504 GB/s of effective memory bandwidth!

When plugged into the full engine forward pass on our H100, end-to-end token generation throughput confirmed the win:

- **Batch Size 1:** Throughput jumped from **66.5 tok/s** in Hugging Face eager to **162.1 tok/s** in QUIC, a **2.44x speedup**.

- **Batch Size 16:** Throughput jumped from **716.8 tok/s** in Hugging Face eager to **1876.4 tok/s** in QUIC, a **2.62x speedup**.

By slicing the K dimension across SMs and skipping padding tokens at the hardware level, we turned an underutilized 14-block bottleneck into full GPU saturation.

<hr>

---

## Quantization: From Mathematical Foundations to Native W4A16 Execution

Quantization sounds like a free lunch on paper: compress your weights from 16-bit floats down to 4-bit integers, get a 4x smaller model, cut memory traffic by 75%, and watch your engine run twice as fast. When I started testing this, first on my local GTX 1650 and later on an H100, I expected an instant doubling in speed. Instead, I hit the exact trap that catches almost everyone experimenting with quantized LLMs.

In this section, we take a deep dive into the engineering and mathematical realities of post-training quantization: from linear mapping constraints and zero-point padding safety, to the theoretical breakthrough of **Activation-Aware Weight Quantization (AWQ)**, our from-scratch quantizer that outperformed the official AutoAWQ library on academic benchmarks, and the low-level CUDA kernels (Marlin and Machete) that execute packed 4-bit GEMM on modern NVIDIA GPUs.

### 1. Demystifying Quantization: Representation vs Hardware Arithmetic

A common misconception is that quantization simply means *"representing weights in integers instead of floats to do integer math."* While integer operations are computationally simpler at the transistor level—a standard 2's complement integer addition with carry executes in a single clock cycle, whereas IEEE 754 floating-point operations require 4 to 8 cycles for exponent alignment and mantissa normalization—this is not how post-training quantization operates in modern LLM serving.

In state-of-the-art serving systems, weights are quantized to lower precision (such as INT4) primarily as a **storage and memory bandwidth compression format**. The packed 4-bit integers themselves have no direct semantic meaning to the model; they are indices mapped onto a continuous mathematical grid. During inference, we encounter two distinct paradigms:

  - **W4A16 (Weight-Only Quantization):** Weights reside in GPU VRAM as packed 4-bit integers. As they are loaded across the memory bus into registers, a specialized CUDA kernel dynamically dequantizes them back to 16-bit floats on the fly. The matrix multiplication itself ($Y = W \cdot X$) is executed entirely in FP16/BF16 arithmetic on Tensor Cores. The activations ($X$) remain in full 16-bit precision throughout the forward pass, incurring zero activation quantization error.

  - **W4A4 / W8A8 (Weight-and-Activation Quantization):** Both weights and dynamic activations are quantized into integers. Computations occur directly on dedicated hardware integer Tensor Cores (such as DP4A). While this offers substantial compute acceleration at high batch sizes, it introduces significant activation quantization noise and requires runtime overhead to compute activation scales on the fly.

Because single-stream and small-batch decoding is strictly **memory-bandwidth bound** (loading weights from DRAM dominates total execution time), **W4A16 is the optimal sweet spot**: it slashes weight memory traffic by up to 75% without the severe accuracy loss of quantizing dynamic activations.

### 2. Mathematical Formulation: Symmetric vs Asymmetric Mapping

Uniform linear quantization maps continuous high-precision floating-point values $w \in \mathbb{R}$ to discrete lower-precision integer bins $q \in \mathbb{Z}$ using a scale factor $S \in \mathbb{R}^+$ and an optional zero-point offset $z \in \mathbb{Z}$:

<p>$$q = 	ext{clamp}\left(\left\lfloor rac{w}{S} 
ight
ceil + z, \; q_{\min}, \; q_{\max}
ight)$$</p>

To reconstruct the estimated floating-point value $\hat{w} pprox w$ during inference dequantization, we apply the inverse mapping:

$$\hat{w} = S \cdot (q - z)$$

#### A. Symmetric Mapping ($z = 0$)

In symmetric quantization, the zero-point is fixed to zero ($z = 0$). The range is centered symmetrically around zero, mapping $[-a, a] 	o [-2^{b-1}+1, \; 2^{b-1}-1]$ (for signed INT4, typically $[-7, 7]$). While leaving one of the 16 possible values unused, it provides an enormous architectural benefit in CUDA kernels: **all zero-point cross-terms vanish completely**.

When calculating matrix-vector products $Y = \hat{W} X = S(Q - z)X$, setting $z = 0$ simplifies the expression to $Y = S \cdot (Q X)$. The GPU avoids accumulating extra subtraction terms across the inner reduction loop, saving ALU cycles and register pressure.

<h4>B. Asymmetric Mapping ($z 
eq 0$)</h4>
In asymmetric quantization, the scale $S$ and zero-point $z$ span an arbitrary real range $[w_{\min}, w_{\max}]$:

<p>$$S = rac{w_{\max} - w_{\min}}{2^b - 1}, \quad z = 	ext{round}\left(-rac{w_{\min}}{S}
ight) + q_{\min}$$</p>

Asymmetric mapping utilizes all 16 discrete bins (0 to 15 for unsigned INT4), providing higher fidelity for skewed or biased weight distributions.

<blockquote>
  **The Sacred Zero-Point Rule in LLMs:** Even if weight values span strictly positive numbers (e.g., from $+20.0$ to $+80.0$), the quantization range **must be artificially extended to include $0.0$** (mapping $[0.0, 80.0]$). Why? Because $0.0$ in full precision **must map to an exact integer with zero residual error**.

  In modern autoregressive transformers, sequence padding and KV-cache initializations rely universally on exact zeros. If $0.0$ incurs even a fractional rounding error after dequantization, padding tokens cease to be neutral zeros and inject artificial semantic activations into every attention layer, triggering catastrophic numerical drift across the 24 transformer blocks.

</blockquote>

#### The Range vs Precision Trade-off

Total quantization error stems from two competing forces: **clipping error** and **rounding error**.

  - **Increasing the scale $S$:** Expands the dynamic range, preventing large outlier weights from being clipped at $q_{\min}$ or $q_{\max}$. However, it stretches the distance between quantization levels, increasing average rounding error.

  - **Decreasing the scale $S$:** Compresses the quantization step size, yielding finer resolution and minimal rounding error for 99% of normal weights. However, any weight beyond the boundaries is harshly clipped.

### 3. Calibrating the Scale Factor $S$: Min-Max vs MSE Observer

The scale factor $S$ is the most decisive parameter in quantization:

1. **Min-Max Observer:** Sets bounds directly from the absolute minimum and maximum values in the tensor. While trivial, it is highly sensitive to outliers. If 99.9% of weights in a layer lie tightly inside $(-2.0, +1.0)$ but a single outlier spikes at $+5.0$, the entire 16-bin INT4 grid is stretched across $(-2.0, +5.0)$. Over 60% of representation capacity is completely wasted on empty space.
2. **MSE Observer (Our Implementation Choice):** Rather than naively accepting extreme outliers, we determine $S$ by minimizing the Mean Squared Error (MSE) between the original high-precision tensor and its reconstructed counterpart:
    $$S^* = rg\min_{S} \; \mathbb{E}\left[ \|W - \hat{W}(S)\|_2^2 
ight]$$
    We implement this via an iterative grid search: starting from the global Min-Max bounds, we iteratively shrink the dynamic range (contracting by 1% steps over 100 iterations), perform fake-quantization on calibration tensors, and evaluate the resulting MSE. The threshold yielding the minimum reconstruction loss is selected. This mathematical contraction discards benign tail outliers to grant dramatically higher resolution to the dense central mass of weights.
3. **Output (Logit) Observer (GPTQ style):** Minimizes output activation error $\|Y - \hat{Y}\|_2^2$ across inverse Hessian matrices. While effective for domain-specific models, it acts greedily on the calibration dataset and poses substantial risks of task-specific overfitting.

### 4. Activation-Aware Weight Quantization (AWQ): Protecting Salient Weights

Traditional quantization schemes treat all weights within a matrix uniformly. However, research in **AWQ (Activation-aware Weight Quantization)** demonstrated an essential empirical property: **protecting just 0.1% to 1% of salient weight channels from quantization error preserves virtually the entire generative perplexity of the base model!**

In baseline Round-to-Nearest (RTN) quantization without protection, Qwen's perplexity degraded sharply to **23.54**. Leaving a mere 1% of salient channels unquantized brought perplexity down to **11.39**—indistinguishable from unquantized FP16.

#### Why Are Certain Weights More Important?

Consider the linear transformation $Y = W \cdot X$. In weight-only quantization, the output error $\Delta Y$ is:

$$\Delta Y = \Delta W \cdot X = (W - \hat{W}) \cdot X$$

The output error is the product of weight quantization error $\Delta W$ and the **magnitude of the incoming activation $X$**. Even if a weight has a tiny quantization error $\Delta W$, if it is multiplied by an activation channel with large magnitude ($|X| \gg 1$), that error is amplified into a massive output perturbation! Conversely, large quantization errors on channels where activations are near zero have negligible impact on generated logits.

Therefore, **salient weights are simply the weights that interact with large activation channels**. We identify these channels by passing calibration sequences from a general corpus through the model and profiling average activation magnitudes per channel.

#### Per-Channel Protection via Equivalent Scaling

Instead of keeping salient weights in mixed precision (which causes memory layout fragmentation and destroys CUDA SIMD parallelism), AWQ applies an **equivalent mathematical transformation**. We multiply the weight matrix by a per-channel scaling diagonal matrix $s$ and divide activations by the exact same factor:

$$Y = W X = (W \cdot 	ext{diag}(s)) \cdot (	ext{diag}(s)^{-1} X)$$

This formulation produces two coordinated benefits:

1. **Weight Scaling:** Multiplying salient weight channels by $s > 1$ pushes their magnitudes higher relative to the quantization grid, reducing relative discretization error.
2. **Activation Attenuation:** Dividing activations by $s$ shrinks the error multiplier in $\Delta Y = \Delta W \cdot (X / s)$, suppressing output deviation.

#### Zero-Overhead Hardware Execution via Normalization Folding

How do we divide activations by $s$ during high-speed inference without adding an extra CUDA kernel launch? **We fold the scaling vector $s$ directly into the preceding layer normalization!**

In Qwen 2.5, every linear projection (QKV, Gate, Up) is preceded by an RMSNorm layer: $	ext{RMSNorm}(x) = rac{x}{	ext{RMS}(x)} \odot \gamma$. We simply reparameterize the RMSNorm weight vector $\gamma$ ahead of time:

$$\gamma_{	ext{new}} = rac{\gamma_{	ext{old}}}{s}$$

During live GPU inference, activation division happens for free inside the existing RMSNorm kernel. **Zero new kernels, zero memory roundtrips, and zero added latency!**

#### Grid Search Optimization for $s$

To find the optimal scaling vector $s$, we define $s$ as a function of activation magnitudes $s_X$ and weight magnitudes $s_W$ parameterized by $lpha$:

$$s = s_X^lpha / s_W^{1-lpha}$$

We execute a 20-point grid search over $lpha \in [0, 1]$. For each candidate $lpha$, we fake-quantize the scaled weights and measure the output MSE of the transformer block. The candidate $lpha$ that yields the lowest output distortion is permanently baked into the model checkpoint.

### 5. Granularity Spectrum: Per-Channel vs Per-Group (Group 32 vs 128)

In matrix multiplication, each row of the weight matrix $W$ computes one scalar output feature. Applying a single scale factor $S$ across the entire matrix (Per-Tensor) yields disastrous accuracy (perplexity shoots above 60+). Applying scales **Per-Channel (Per-Row)** is a massive improvement because each row maintains its own independent dynamic range, keeping the inner product a pure streaming Multiply-Accumulate (MAC) pipeline.

However, when a row contains thousands of elements, a single scale factor can still be too coarse. We divide each row into contiguous **groups** of size $G$ (typically $G = 32$ or $G = 128$):

  - **Group Size = 32:** Adapts tightly to local weight fluctuations. It delivers superior accuracy and lower perplexity, but requires storing $4	imes$ more scale metadata and interrupts the continuous hardware MAC loop more frequently, incurring a small latency penalty.

  - **Group Size = 128:** The industry standard for high-performance inference. It strikes the perfect balance: scale metadata adds less than 3% storage overhead, and the 128-element chunks align seamlessly with modern Tensor Core memory tile layouts (such as Marlin's 16x64 thread blocks).

### 6. Our From-Scratch Quantizer Outperforms the Official AWQ Library

Rather than simply running a generic third-party script, we engineered our complete AWQ quantization engine **from scratch**. We implemented the calibration profiling pipeline, custom MSE grid search, asymmetric integer packing, and Marlin-compatible weight reordering.

To rigorously validate our implementation, we evaluated our quantized Qwen2.5-0.5B checkpoint against both the unquantized base model and the official AutoAWQ library across four standard LLM benchmarks: **MMLU** (general reasoning), **C-Eval** (multi-discipline knowledge), **IFEval** (instruction following), and **Perplexity at 1024 context**.

<table>
  <thead>
    <tr>
      <th>Evaluation Setup</th>
      <th>Model Size (MB)</th>
      <th>Latency (ms)</th>
      <th>MMLU (%)</th>
      <th>C-Eval (%)</th>
      <th>IFEval (%)</th>
      <th>PPL @ 1024 (Lower is Better)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>**Dense Base Model (BF16)**</td>
      <td>942.30 MB</td>
      <td>197.12 ms</td>
      <td>39.65%</td>
      <td>39.67%</td>
      <td>24.12%</td>
      <td>**16.01**</td>
    </tr>
    <tr>
      <td>**Quantized with Official AWQ Library (G=128)**</td>
      <td>437.20 MB</td>
      <td>123.50 ms</td>
      <td>29.93%</td>
      <td>33.71%</td>
      <td>20.14%</td>
      <td>18.79</td>
    </tr>
    <tr>
      <td>**Quantized From Scratch (Our AWQ Engine, G=128)**</td>
      <td>437.20 MB</td>
      <td>123.50 ms</td>
      <td>**32.05%**</td>
      <td>**34.80%**</td>
      <td>**21.14%**</td>
      <td>**18.77**</td>
    </tr>
    <tr>
      <td>**Quantized From Scratch (Our AWQ Engine, G=32)**</td>
      <td>457.10 MB</td>
      <td>136.30 ms</td>
      <td>**35.12%**</td>
      <td>**37.15%**</td>
      <td>**21.14%**</td>
      <td>**17.41**</td>
    </tr>
  </tbody>
</table>

The empirical data reveals a clear and decisive outcome: **Our custom-built quantization pipeline outperformed the official AWQ library across every single evaluation metric**:

  - On **MMLU**, our engine scored **32.05% vs 29.93%** (+2.12% higher accuracy).

  - On **C-Eval**, our engine scored **34.80% vs 33.71%** (+1.09% higher accuracy).

  - On **IFEval**, our engine scored **21.14% vs 20.14%** (+1.00% better instruction fidelity).

  - On **Perplexity**, our engine achieved **18.77 vs 18.79**.

Furthermore, testing Group Size 32 demonstrated that tighter grouping pushes MMLU all the way to **35.12%** and Perplexity down to **17.41**, within striking distance of the uncompressed BF16 baseline while cutting latency from 197.1 ms to 136.3 ms.

![Empirical Benchmark Comparison: Base Model vs Official AWQ Library vs Our Custom From-Scratch Quantizer](images/custom_quantizer_scratch_vs_official_awq.png)
*Empirical Benchmark Comparison: Base Model vs Official AWQ Library vs Our Custom From-Scratch Quantizer*

Figure: Group Size = 128 benchmark comparison demonstrating our from-scratch implementation beating the official AWQ library across MMLU, C-Eval, and IFEval.

![Group Size 32 Evaluation Table](images/custom_awq_group32_benchmark_table.png)
*Group Size 32 Evaluation Table*

Figure: Group Size = 32 benchmark results confirming tighter accuracy retention at the cost of a minor 20 MB metadata overhead.

### 7. The Embedding Matrix Dilemma: Why 0.5B Compresses by 50% Instead of 75%

When compressing a model with 4-bit weights, basic arithmetic suggests the checkpoint should shrink to 25% of its original size. Yet when we quantize Qwen 2.5 0.5B, the disk footprint drops from **942.3 MB down to 457.1 MB**—a reduction of approximately **51.5%**, not 75%. Why?

The answer lies in the **vocabulary embedding matrix (`embed_tokens`)**. Qwen 2.5 employs an expansive multilingual vocabulary of **151,936 tokens** with a hidden dimension of 896. In FP16, this single weight matrix consumes:

$$151,936 	imes 896 	imes 2 	ext{ bytes} pprox 272.3 	ext{ MB}$$

In a small 494M parameter model, the embedding matrix alone represents **more than 27% of the entire model's parameters**! After quantizing all 24 transformer blocks down to 4-bit, the linear layers shrink to roughly 195 MB, while the embedding table remains unquantized in 16-bit at ~260 MB (tied between input embeddings and the LM head).

![Storage Share by Precision for Qwen2.5-0.5B Quantized Model](images/qwen2_5_0_5b_quantized_storage_breakdown.png)
*Storage Share by Precision for Qwen2.5-0.5B Quantized Model*

Figure: Storage breakdown of Qwen2.5-0.5B: unquantized 16-bit embeddings account for 61.5% of the total quantized model size.

#### Why Not Quantize Embeddings to 4-Bit?

The embedding table is the very first operator the input sees and the exact matrix the final hidden state multiplies against to generate vocabulary logits. Because it maps discrete one-hot token IDs into dense semantic representations, quantizing it to INT4 introduces severe linguistic degradation and perplexity explosion. Virtually all production engines leave embeddings in 16-bit precision. By explicitly **tying the input embedding weights to the output LM head**, we avoid storing duplicate 260 MB matrices, keeping total memory at 457 MB.

#### Does This Overhead Persist on Larger Models?

To verify our architectural hypothesis, we performed the identical storage analysis on **Qwen-1.5B**:

![Storage Share by Precision for Qwen-1.5B Quantized Model](images/qwen2_1_5b_quantized_storage_breakdown.png)
*Storage Share by Precision for Qwen-1.5B Quantized Model*

Figure: Storage breakdown of Qwen-1.5B: quantized INT4 linear layers expand to 57.5%, driving overall size down to 33% of the original model.

Because transformer depth and feed-forward intermediate dimensions scale quadratically while vocabulary size remains constant, the embedding matrix accounts for a rapidly shrinking fraction of total parameters as models grow. For 1.5B, the model compresses to **33%** of its original size; for 7B and 70B, it achieves the near-ideal **26% to 27%** footprint.

### 8. Escaping the CPU Dequantization Trap

Once you have a pristine 4-bit checkpoint, the next hurdle is deployment. When I first wired our AWQ model into a naive PyTorch pipeline, I was greeted by a maddening result: **GPU memory usage remained at 1.5 GB (identical to FP16), and generation was actually slower!**

![The CPU Dequantization Trap Diagram](images/cpu-dequantization-trap-memory-bottleneck.png)
*The CPU Dequantization Trap Diagram*

Figure: The CPU Dequantization Trap: dequantizing weights in host RAM transfers bloated 16-bit matrices over PCIe, burning CPU cycles and nullifying GPU memory savings.

In standard high-level implementations, quantized weights are loaded from disk into CPU memory. The framework then immediately dequantizes them in CPU RAM back to FP16, and transfers bloated 16-bit matrices across the PCIe bus into GPU VRAM! You get zero memory savings on the GPU, zero bandwidth reduction across DRAM, and you waste CPU cycles doing point-wise float conversions.

![Empirical VRAM comparison proving naive quantization fails](images/naive-awq-quantization-vram-empirical-proof.png)
*Empirical VRAM comparison proving naive quantization fails*

Figure: Empirical proof: Across all context lengths, naive execution exhibits 100% identical VRAM usage to stock FP16.

If weights do not stay packed as 4-bit integers all the way across the GPU memory bus into the SM registers, quantization is completely pointless.

### 9. Native W4A16 GPU Kernels: Marlin and Hopper Machete

To realize true hardware acceleration, weights must remain packed as 4-bit nibbles in VRAM. The GPU streams packed INT4 bytes over high-bandwidth memory (HBM), unrolls them on the fly inside registers or shared memory, and executes floating-point matrix multiplications with 16-bit activations.

Look at the memory traffic savings during single-token ($M=1$) decode on an H100:

<table>
  <thead>
    <tr>
      <th>Projection Layer</th>
      <th>M</th>
      <th>Latency</th>
      <th>TFLOPS</th>
      <th>FP16 Data</th>
      <th>Bandwidth</th>
      <th>INT4 Data</th>
      <th>Traffic Reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>`q_proj`</td>
      <td>1</td>
      <td>0.0134 ms</td>
      <td>0.12</td>
      <td>1.61 MB</td>
      <td>120.3 GB/s</td>
      <td>0.47 MB</td>
      <td>**3.45x fewer bytes**</td>
    </tr>
    <tr>
      <td>`k_proj`</td>
      <td>1</td>
      <td>0.0125 ms</td>
      <td>0.02</td>
      <td>0.23 MB</td>
      <td>18.4 GB/s</td>
      <td>0.07 MB</td>
      <td>**3.45x fewer bytes**</td>
    </tr>
    <tr>
      <td>`gate_proj`</td>
      <td>1</td>
      <td>0.0092 ms</td>
      <td>0.94</td>
      <td>8.72 MB</td>
      <td>943.8 GB/s</td>
      <td>2.52 MB</td>
      <td>**3.46x fewer bytes**</td>
    </tr>
    <tr>
      <td>`down_proj`</td>
      <td>1</td>
      <td>0.0117 ms</td>
      <td>0.74</td>
      <td>8.72 MB</td>
      <td>743.9 GB/s</td>
      <td>2.52 MB</td>
      <td>**3.46x fewer bytes**</td>
    </tr>
  </tbody>
</table>

For large feed-forward layers like `gate_proj` and `down_proj`, bytes transferred per decode step plummet from **8.72 MB down to 2.52 MB**—an instant **3.46x reduction in DRAM traffic**!

We integrated two specialized 4-bit execution backends into QUIC:

#### A. Marlin Kernel (Turing, Ampere, Ada, Hopper)

Marlin is an ultra-fast 4-bit GEMM kernel. In `src/quant/marlin.py`, we take our AWQ checkpoint and repack weights into Marlin's proprietary **16x64 tile layout**. This interleaves weights and scale factors so a single 128-bit vector load instruction populates registers with zero shared memory bank conflicts. In `kernels/cuda/marlin.cu`, it executes asynchronous `mma.sync` Tensor Core instructions with instant on-the-fly dequantization.

#### B. Machete Kernel (NVIDIA Hopper sm_90a)

For NVIDIA Hopper GPUs, CUTLASS 3.x provides hardware-accelerated mixed-input matrix multiplications. In `src/quant/machete.py` and `kernels/cuda/machete.cu`, we prepack weights using `machete_prepack_B`. It executes Warp-Specialized Cooperative WGMMA (Warp Group Matrix Multiply-Accumulate) instructions, asynchronously streaming 16-bit activations and 4-bit weights directly into Hopper tensor cores with FP32 accumulation.

### 10. The 0.5B Trade-off: VRAM Halved, but Decode TPS Slower?

With native W4A16 execution, peak VRAM allocated dropped from **996 MB down to 503 MB**—cutting GPU memory footprint in half! This allows serving double the concurrent batch size on memory-constrained GPUs. However, examining single-stream decode throughput reveals an unexpected phenomenon:

<table>
  <thead>
    <tr>
      <th>Engine & Configuration</th>
      <th>Batch</th>
      <th>Context</th>
      <th>Decode Throughput</th>
      <th>Peak VRAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>**QUIC FP16**</td>
      <td>1</td>
      <td>128</td>
      <td>**162.1 tok/s**</td>
      <td>996 MB</td>
    </tr>
    <tr>
      <td>**QUIC Quant (W4A16 Marlin)**</td>
      <td>1</td>
      <td>128</td>
      <td>148.9 tok/s</td>
      <td>**503 MB**</td>
    </tr>
    <tr>
      <td>**Hugging Face FP16 Eager**</td>
      <td>1</td>
      <td>128</td>
      <td>66.5 tok/s</td>
      <td>1290 MB</td>
    </tr>
  </tbody>
</table>

While QUIC Quant still outperforms Hugging Face by **2.24x**, it is slightly slower than our own bare-metal FP16 engine (148.9 vs 162.1 tok/s). Why does 4-bit quantization accelerate a 70B model by 3x, but slightly decelerate a 0.5B model at Batch 1?

This highlights the fundamental difference between small and massive models:

  - **On a 70B parameter model:** Weight matrices are gargantuan ($8192 	imes 8192$). Moving bytes across the DRAM bus is the single overwhelming bottleneck; memory traffic completely saturates the bus. Cutting traffic by 4x instantly yields a massive 2x to 3x speedup.

  - **On Qwen 2.5 0.5B:** The hidden dimension is only 896. Weight matrices are so tiny that the absolute time spent moving bytes is already negligible. In this regime, the **ALU instruction overhead required to unpack 4-bit nibbles, apply zero-point shifts, and multiply scales inside the CUDA registers** takes more time than the microsecond saved on the memory bus!

On small architectures like 0.5B, **quantization is primarily a VRAM capacity and concurrent throughput multiplier** (allowing massive batch sizes and extended context windows) rather than an isolated single-stream latency booster.

---

## System-Level Serving: Dynamic KV-Cache and Continuous Batching

Writing fast CUDA kernels is only half the battle. In a real serving engine, users do not submit neat, synchronized batches of identical prompts. One user sends a 10-word question; another sends a 1,000-word essay. One request needs 15 tokens; another needs 500 tokens.

If your serving engine handles batches naively, your GPU silicon spends most of its time waiting around on bubbles, no matter how fast your kernels are. Here is how I moved QUIC from an offline generation script into a real serving engine with dynamic KV-caching and Continuous Batching.

### The KV-Cache Problem: Dynamic Slot Allocation

In autoregressive generation, each new token only needs to attend to previous tokens. Recomputing previous Key and Value vectors on every step would turn generation from linear order into quadratic order, an exponential disaster.

So we store all past Key and Value activations in a **KV-Cache**.

In stock PyTorch implementations, managing the KV-cache is sloppy:

- Tensors are dynamically resized with `torch.cat` on every single token.

- Each `cat` triggers a new memory allocation on the GPU heap and copies existing keys and values to new addresses.

In QUIC inside `src/engine.py`, we use structured dynamic slot allocation with `DynamicCache` and GQA indexing:

- Memory for new keys and values is appended directly into pre-allocated layer slots.

- For GQA with 2 KV heads, we maintain clean `[B, 2, S_kv, 64]` tensors without redundant head duplication.

### Static Batching vs Continuous Batching: The Head-of-Line Blocking Trap

To understand why Continuous Batching is necessary, look at what happens in **Static Batching** through `StaticAdmissionPolicy` in `src/scheduler.py`:

<p>

![Static Batching Head-of-Line Blocking and Idle Bubble Waste During Variable-Length Serving](images/static-batching-head-of-line-blocking.jpeg)
*Static Batching Head-of-Line Blocking and Idle Bubble Waste During Variable-Length Serving*

</p>
Suppose you have a maximum batch size of 2, and two requests arrive:

- **Request A:** Prompt: 50 tokens. Needs **10 tokens** of output.

- **Request B:** Prompt: 50 tokens. Needs **200 tokens** of output.

Under static batching:

1. Both requests are admitted together.
2. At token iteration 10, Request A hits `&#x3C;|im_end|>` and finishes.
3. But the batch cannot finish until Request B is done!
4. For the next **190 iterations**, Slot A sits completely empty. The GPU continues running decode kernels for a batch size of 2, but half the compute is purely wasted padding!

This is called **Head-of-Line Blocking**. The slowest request in the batch holds the entire engine hostage.

### Implementing Continuous Batching in QUIC

<p>

![Continuous Batching Iteration-Level Request Scheduling and Dynamic KV-Cache Slot Reallocation](images/continuous-batching-iteration-level-scheduling.jpeg)
*Continuous Batching Iteration-Level Request Scheduling and Dynamic KV-Cache Slot Reallocation*

</p>
Continuous Batching, pioneered by Orca and popularized by vLLM, solves this by scheduling at the **token level** rather than the request level.

In `src/scheduler.py`, the `ContinuousAdmissionPolicy` runs a scheduling loop before every single decode step:

```
class ContinuousAdmissionPolicy:
    def decide(self, running: list[SequenceState], waiting: deque[SequenceState], max_batch_size: int) -> ScheduleDecision:
        # 1. Check which running sequences have finished
        to_release = [seq for seq in running if seq.is_finished]
        
        # 2. Calculate newly available slots
        available_slots = max_batch_size - (len(running) - len(to_release))
        
        # 3. Immediately admit waiting requests into vacant slots
        to_admit = []
        while waiting and len(to_admit) &#x3C; available_slots:
            to_admit.append(waiting.popleft())
            
        return ScheduleDecision(admit=to_admit, release=to_release)
```

Here is how the execution loop in `src/engine.py` operates:

1. **Per-Token Check:** As soon as Request A emits `&#x3C;|im_end|>` at token 151645 or hits its `max_tokens`, it is evicted immediately.
2. **Immediate Admission:** A waiting Request C from the FIFO queue is admitted into Slot A right away.
3. **Prefill and Decode Coexistence:** Request C runs its prompt prefill, populates its KV cache slots, and on the very next token iteration, it joins Request B in the lock-step decode loop.

Zero wasted bubbles. Every slot in the batch does useful work on every cycle.

### Serving Benchmarks: Static vs Continuous

We ran a serving benchmark on our H100 with 8 concurrent requests of varying prompt and generation lengths:

<table><thead><tr><th align="left">Policy</th><th align="center">Max Batch Size</th><th align="center">Requests</th><th align="center">Total Tokens</th><th align="center">Latency</th><th align="center">Serving Throughput</th><th align="center">Peak VRAM</th></tr></thead><tbody><tr><td align="left">**Static Batching**</td><td align="center">4</td><td align="center">8</td><td align="center">256</td><td align="center">2.05s</td><td align="center">**124.6 tok/s**</td><td align="center">1,548 MB</td></tr><tr><td align="left">**Continuous Batching**</td><td align="center">4</td><td align="center">8</td><td align="center">256</td><td align="center">1.77s</td><td align="center">**144.2 tok/s**</td><td align="center">1,549 MB</td></tr></tbody></table>
Continuous batching delivered a **+15.7 percent throughput increase** on the exact same hardware and model weights, purely by eliminating idle slots and overlapping request lifecycles.

<hr>

---

## End-to-End Benchmarks, Profiler Deep-Dive, and Accuracy Parity

After fusing kernels, wiring native 4-bit quantization, and implementing continuous batching, it was time to run the complete benchmark sweep against Hugging Face.

All benchmarks were collected on an **NVIDIA H100 80GB HBM3 GPU** via Modal, sweeping across:

- **Batch Sizes:** `1`, `4`, `16`

- **Context Lengths:** `128`, `512`, `1024`, `2048`

<li>**4 Execution Paths:**

</li>

### The Headline Numbers: 2.62x Higher Decode Throughput

Here are the head-to-head results from our H100 runs:

<table><thead><tr><th align="center">Batch</th><th align="center">Context</th><th align="center">HF Eager FP16</th><th align="center">QUIC Quant W4A16</th><th align="center">QUIC FP16</th><th align="center">QUIC Speedup vs HF</th></tr></thead><tbody><tr><td align="center">**1**</td><td align="center">128</td><td align="center">66.5 tok/s</td><td align="center">148.9 tok/s</td><td align="center">**162.1 tok/s**</td><td align="center">**2.44x**</td></tr><tr><td align="center">**1**</td><td align="center">512</td><td align="center">65.2 tok/s</td><td align="center">144.1 tok/s</td><td align="center">**158.4 tok/s**</td><td align="center">**2.43x**</td></tr><tr><td align="center">**1**</td><td align="center">1024</td><td align="center">63.8 tok/s</td><td align="center">139.6 tok/s</td><td align="center">**153.7 tok/s**</td><td align="center">**2.41x**</td></tr><tr><td align="center">**1**</td><td align="center">2048</td><td align="center">58.4 tok/s</td><td align="center">128.2 tok/s</td><td align="center">**141.9 tok/s**</td><td align="center">**2.43x**</td></tr><tr><td align="center">**4**</td><td align="center">128</td><td align="center">235.1 tok/s</td><td align="center">534.8 tok/s</td><td align="center">**587.2 tok/s**</td><td align="center">**2.50x**</td></tr><tr><td align="center">**4**</td><td align="center">1024</td><td align="center">224.6 tok/s</td><td align="center">498.1 tok/s</td><td align="center">**551.0 tok/s**</td><td align="center">**2.45x**</td></tr><tr><td align="center">**16**</td><td align="center">128</td><td align="center">716.8 tok/s</td><td align="center">1,692.4 tok/s</td><td align="center">**1,876.4 tok/s**</td><td align="center">**2.62x**</td></tr><tr><td align="center">**16**</td><td align="center">1024</td><td align="center">682.3 tok/s</td><td align="center">1,581.1 tok/s</td><td align="center">**1,744.2 tok/s**</td><td align="center">**2.56x**</td></tr></tbody></table>
**TTFT and Memory Footprint:**

- **TTFT, Time To First Token:** QUIC consistently beats Hugging Face across all context lengths. For example at Context 128, TTFT is **14.1 ms vs 19.8 ms**, and at Context 2048, it reaches **35.8 ms vs 48.7 ms**.

- **Peak VRAM:** `quic_quant` holds memory strictly at **~780 MB** across all sequence lengths, cutting the FP16 memory footprint of **1,548 MB** right in half.

### Profiler Timeline Deep-Dive: Where Hugging Face Lost the Race

To see exactly where Hugging Face bled performance, I set up a profiling run using `torch.profiler` on the NVIDIA H100.

Both models were warmed up through 1 prefill step and 10 decode steps to prime CUDA memory caches and stabilize the driver. Then the profiler captured 1 prefill step followed by 5 decode steps at Batch Size 1 and Context Length 512.

The exported Chrome traces are available in the repository assets:

- Hugging Face Trace: `images/hf_trace.json.gz`

- QUIC Engine Trace: `images/quic_trace.json.gz`

Opening these traces side-by-side in `ui.perfetto.dev` reveals what happens inside the execution stream:

#### Top Hugging Face Kernels by Duration

<table><thead><tr><th align="left">Kernel Name</th><th align="left">Launch Count</th><th align="left">Total GPU Time</th><th align="left">Share of CUDA Time</th></tr></thead><tbody><tr><td align="left">`nvjet_hsh_64x8_64x16_4x1_v_bz_TNT` GEMM</td><td align="left">360</td><td align="left">1.81 ms</td><td align="left">10.3 percent</td></tr><tr><td align="left">`elementwise_kernel MulFunctor`</td><td align="left">625</td><td align="left">1.21 ms</td><td align="left">6.9 percent</td></tr><tr><td align="left">`unrolled_elementwise_kernel direct_copy`</td><td align="left">300</td><td align="left">0.98 ms</td><td align="left">5.6 percent</td></tr><tr><td align="left">`nvjet_hsh_64x8_64x16_2x1_v_bz_bias_TNT`</td><td align="left">240</td><td align="left">0.93 ms</td><td align="left">5.3 percent</td></tr><tr><td align="left">`flash_fwd_splitkv_kernel` Attention</td><td align="left">120</td><td align="left">0.81 ms</td><td align="left">4.6 percent</td></tr><tr><td align="left">`vectorized_elementwise_kernel CUDAFunctor_add`</td><td align="left">528</td><td align="left">0.77 ms</td><td align="left">4.4 percent</td></tr><tr><td align="left">`reduce_kernel MeanOps` RMSNorm Reductions</td><td align="left">294</td><td align="left">0.74 ms</td><td align="left">4.2 percent</td></tr><tr><td align="left">`CatArrayBatchedCopy` KV Cache Copies</td><td align="left">336</td><td align="left">0.74 ms</td><td align="left">4.2 percent</td></tr></tbody></table>
Notice that un-fused elementwise multiplications, unrolled copies, adds, and reduction kernels consumed over 30 percent of Hugging Face compute time and thousands of launch queues!

#### Top QUIC Kernels by Duration

<table><thead><tr><th align="left">Kernel Name</th><th align="left">Launch Count</th><th align="left">Total GPU Time</th><th align="left">Share of CUDA Time</th></tr></thead><tbody><tr><td align="left">`nvjet_hsh_64x8_64x16_4x1_v_bz_TNT` GEMM</td><td align="left">480</td><td align="left">2.25 ms</td><td align="left">20.2 percent</td></tr><tr><td align="left">`fad_decode_kernel` Split-K Decode Attention</td><td align="left">120</td><td align="left">0.87 ms</td><td align="left">7.8 percent</td></tr><tr><td align="left">`nvjet_hsh_64x8_64x16_2x1_v_bz_TNT` GEMM</td><td align="left">240</td><td align="left">0.84 ms</td><td align="left">7.5 percent</td></tr><tr><td align="left">`nvjet_hsh_64x8_64x16_4x1_v_bz_splitK_TNT`</td><td align="left">120</td><td align="left">0.70 ms</td><td align="left">6.3 percent</td></tr><tr><td align="left">`gemm_d::add_bias_kernel` Fused Bias GEMM</td><td align="left">432</td><td align="left">0.62 ms</td><td align="left">5.6 percent</td></tr><tr><td align="left">`qwen2_rmsnorm_kernel` Fused RMSNorm</td><td align="left">294</td><td align="left">0.55 ms</td><td align="left">4.9 percent</td></tr><tr><td align="left">`fad_combine_kernel` Split-K Reductions</td><td align="left">120</td><td align="left">0.41 ms</td><td align="left">3.6 percent</td></tr><tr><td align="left">`qwen2_rope_kernel` Fused RoPE</td><td align="left">288</td><td align="left">0.41 ms</td><td align="left">3.6 percent</td></tr><tr><td align="left">`swiglu_naive_half_kernel` Fused SwiGLU</td><td align="left">144</td><td align="left">0.32 ms</td><td align="left">2.8 percent</td></tr></tbody></table>
In QUIC, the micro-kernel storm is completely gone:

- The 625 separate `MulFunctor` calls and 294 separate `MeanOps` reductions were replaced by `qwen2_rmsnorm_kernel`, running in a single vectorized pass per layer.

- Decode attention ran through our dedicated `fad_decode_kernel`, chunking the KV history across 64-token splits and keeping all 132 SMs active.

- Dispatch idle bubbles were cut by over 101 milliseconds, collapsing total span from 185.58 ms down to 77.20 ms.

### Strict Parity: Does It Actually Generate the Same Words?

Speed is meaningless if numerical drift turns your model into a random token generator.

Because we wrote custom RMSNorm, RoPE, SwiGLU, and Split-K attention kernels, minor floating-point rounding differences across 24 sequential transformer layers could easily accumulate.

In `tests/test_parity.py`, I set up two rigorous validation checks:

#### 1. Logits Tolerance Check

We feed an identical prompt into both Hugging Face and QUIC, and measure the difference between their raw logits before softmax:

- **Mean Absolute Difference:** ~0.011

- **Max Absolute Difference:** &#x3C; 0.42

- **Argmax Agreement:** Exactly 100 percent agreement on the predicted next token ID.

#### 2. The 100-Token Word-for-Word Test

Then comes the real stress test: generate a 100-token response from scratch on an open-ended physics prompt:

`"Explain the fundamental principles of physics and how gravity works:"`

We run both engines in greedy mode for exactly 100 tokens, decode them, and compare:

```
assert quic_generated == hf_generated, "Token mismatch detected!"
assert quic_text == hf_text, "Decoded text does not match word-for-word!"
```

**Result: 100 percent word-for-word identity.** Every single punctuation mark, space, and token matches Hugging Face exactly, while generating at over **2.5x the speed**.

### What I Learned and What’s Next

Building an inference engine from bare metal teaches lessons you never grasp from high-level frameworks:

1. **Memory bandwidth is the real enemy:** During decode, your peak FLOPS mean nothing if you launch hundreds of un-fused kernels per token. Fusing RMSNorm, RoPE, and SwiGLU is where the immediate wins live.
2. **Attention must be split during decode:** On large GPUs like the NVIDIA H100 with 132 SMs, standard decode attention leaves over 85 percent of your silicon starving at small batch sizes. Split-K is mandatory to keep hardware saturated.
3. **Quantization behaves differently on small models:** On a 70B parameter model, 4-bit quantization yields massive throughput acceleration. On a 0.5B model, unpacking ALU overhead slightly offsets bandwidth gains, making it a VRAM capacity optimization rather than a pure latency booster.
4. **Serving is about slot utilization:** Continuous batching proves that systems-level scheduling delivers a free 15 to 20 percent throughput boost without touching CUDA instructions.

All code, CUDA kernels, tests, traces, and benchmark scripts are open-source on GitHub at [CaptainLevi04/quic.cu](https://github.com/CaptainLevi04/quic.cu).

Feel free to check out the repository, run the benchmarks yourself, and inspect the traces in Perfetto.

### Team Contributions & Project Acknowledgements

QUIC was built as an end-to-end graduation project by a dedicated team of engineers who divided responsibilities across architectural layers:

  - **Amr Gamal ([@CaptainLevi04](https://github.com/CaptainLevi04)):** Project Lead & Quantization Architecture. Engineered the from-scratch AWQ Quantizer (outperforming AutoAWQ), released the Hugging Face model checkpoint, conducted embedding matrix storage & memory breakdown analyses, and directed evaluation sweeps.

  - **Hassan Dahroug ([@hdahroug](https://github.com/hdahroug)):** Custom CUDA Kernels. Authored the hand-written CUDA workhorses including vectorized RMSNorm, register-resident RoPE, fused SwiGLU, FlashAttention prefill, and Split-K GQA decode attention.

  - **Islam Essam ([@IslamEssam01](https://github.com/IslamEssam01)):** System-Level Serving. Developed the serving runtime, dynamic KV-cache slot allocation, and the iteration-level continuous batching scheduler.

### Citation

If you find QUIC or our quantization research useful, please cite:

```
@software{quic2026,
  author    = {Gamal, Amr and Dahroug, Hassan and Essam, Islam},
  title     = {QUIC: Qwen Unified Inference on CUDA},
  year      = {2026},
  url       = {https://github.com/CaptainLevi04/quic.cu}
}
```

 <hr class="my-8 border-dashed">  
  Connect & Project Links:

  
    <a href="https://github.com/CaptainLevi04/quic.cu" title="Project Repository" class="group inline-flex items-center gap-1 hover:text-accent p-2 sm:p-1">
      GitHub: CaptainLevi04/quic.cu
    </a>
    •
    <a href="https://huggingface.co/Amr04/Qwen2.5-0.5B-AWQ-W4A16-tied-embed-asym-hq128" title="Hugging Face Model" class="group inline-flex items-center gap-1 hover:text-accent p-2 sm:p-1">
      Hugging Face: Qwen2.5-0.5B-AWQ-W4A16
    </a>
    •
    <a href="https://github.com/CaptainLevi04" title="Amr Gamal on GitHub" class="group inline-flex items-center gap-1 hover:text-accent p-2 sm:p-1">
      Author: Amr Gamal
    </a>
  

  <!-- Load full theme logic --> <script type="module">const s="theme",c="light",a="dark",i="light";function l(){const e=localStorage.getItem(s);return e||i}let t=window.theme?.themeValue??l();function r(){localStorage.setItem(s,t),o()}function o(){document.firstElementChild?.setAttribute("data-theme",t),document.querySelector("#theme-btn")?.setAttribute("aria-label",t);const e=document.body;if(e){const n=window.getComputedStyle(e).backgroundColor;document.querySelector("meta[name='theme-color']")?.setAttribute("content",n)}}window.theme?(window.theme.setPreference=r,window.theme.reflectPreference=o):window.theme={themeValue:t,setPreference:r,reflectPreference:o,getTheme:()=>t,setTheme:e=>{t=e}};o();function d(){o(),document.querySelector("#theme-btn")?.addEventListener("click",()=>{t=t===c?a:c,window.theme?.setTheme(t),r()})}d();document.addEventListener("astro:after-swap",d);document.addEventListener("astro:before-swap",e=>{const m=e,n=document.querySelector("meta[name='theme-color']")?.getAttribute("content");n&&m.newDocument.querySelector("meta[name='theme-color']")?.setAttribute("content",n)});window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",({matches:e})=>{t=e?a:c,window.theme?.setTheme(t),r()});</script> </body> </html> <script data-astro-rerun>
  /** Create a progress indicator
   *  at the top */
  function createProgressBar() {
    // Create the main container div
    const progressContainer = document.createElement("div");
    progressContainer.className =
      "progress-container fixed top-0 z-10 h-1 w-full bg-background";

    // Create the progress bar div
    const progressBar = document.createElement("div");
    progressBar.className = "progress-bar h-1 w-0 bg-accent";
    progressBar.id = "myBar";

    // Append the progress bar to the progress container
    progressContainer.appendChild(progressBar);

    // Append the progress container to the document body or any other desired parent element
    document.body.appendChild(progressContainer);
  }
  createProgressBar();

  /** Update the progress bar
   *  when user scrolls */
  function updateScrollProgress() {
    document.addEventListener("scroll", () => {
      const winScroll =
        document.body.scrollTop || document.documentElement.scrollTop;
      const height =
        document.documentElement.scrollHeight -
        document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      if (document) {
        const myBar = document.getElementById("myBar");
        if (myBar) {
          myBar.style.width = scrolled + "%";
        }
      }
    });
  }
  updateScrollProgress();

  /* Go to page start after page swap */
  document.addEventListener("astro:after-swap", () =>
    window.scrollTo({ left: 0, top: 0, behavior: "instant" })
  );
</script>