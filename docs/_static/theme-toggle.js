/* A two-state light/dark button: sun or moon, tooltip, one click
   toggles. Writes the same localStorage keys and <html> attributes
   the theme itself uses, so everything styled by data-theme follows. */
document.addEventListener("DOMContentLoaded", function () {
  var stock = document.querySelector("button.theme-switch-button");
  if (!stock || !stock.parentNode) {
    return;
  }

  function resolved() {
    var t = document.documentElement.dataset.theme;
    if (t === "dark" || t === "light") {
      return t;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark" : "light";
  }

  function apply(theme) {
    document.documentElement.dataset.mode = theme;
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("mode", theme);
      localStorage.setItem("theme", theme);
    } catch (e) { /* storage unavailable: still toggles this page */ }
  }

  var btn = document.createElement("button");
  btn.className = "btn btn-sm nav-link pst-navbar-icon";
  var icon = document.createElement("i");
  icon.className = "fa-solid fa-lg";
  btn.appendChild(icon);

  function paint() {
    var dark = resolved() === "dark";
    icon.classList.toggle("fa-moon", !dark);
    icon.classList.toggle("fa-sun", dark);
    var next = dark ? "light" : "dark";
    btn.title = "Switch to " + next + " mode";
    btn.setAttribute("aria-label", btn.title);
  }

  btn.addEventListener("click", function () {
    apply(resolved() === "dark" ? "light" : "dark");
    paint();
  });

  paint();
  stock.parentNode.insertBefore(btn, stock);
});
