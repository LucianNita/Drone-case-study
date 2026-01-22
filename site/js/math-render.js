document.addEventListener("DOMContentLoaded", function() {
  if (typeof renderMathInElement !== "function") {
    console.warn("KaTeX auto-render not found; math will not be rendered.");
    return;
  }
  renderMathInElement(document.body, {
    // keep the default behavior: $$...$$ for display, $...$ for inline
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "$", right: "$", display: false}
    ],
    // ignore code blocks and preformatted content
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
});