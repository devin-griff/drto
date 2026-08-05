/* A two-state light/dark button: the moon shows in light mode, the
   sun in dark mode (CSS on html[data-theme] picks the icon), and one
   click toggles. No tooltip; the aria-label serves screen readers.
   Writes the same localStorage keys and <html> attributes the theme
   itself uses, so everything styled by data-theme follows. The stock
   switchers (a three-mode cycler or a dropdown, depending on theme
   version) are hidden with inline important styles, which no theme
   stylesheet can override. */
document.addEventListener("DOMContentLoaded", function () {
  var stocks = document.querySelectorAll(".theme-switch-button");
  if (!stocks.length) {
    return;
  }
  stocks.forEach(function (el) {
    el.style.setProperty("display", "none", "important");
    var menu = el.nextElementSibling;
    if (menu && menu.classList.contains("dropdown-menu")) {
      menu.style.setProperty("display", "none", "important");
    }
  });

  function apply(theme) {
    document.documentElement.dataset.mode = theme;
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("mode", theme);
      localStorage.setItem("theme", theme);
    } catch (e) { /* storage unavailable: still toggles this page */ }
  }

  var btn = document.createElement("button");
  btn.id = "drto-theme-toggle";
  btn.className = "btn btn-sm nav-link pst-navbar-icon";
  btn.setAttribute("aria-label", "Toggle light and dark mode");
  btn.innerHTML =
    '<i class="fa-solid fa-moon fa-lg"></i>' +
    '<i class="fa-solid fa-sun fa-lg"></i>';

  btn.addEventListener("click", function () {
    var dark = document.documentElement.dataset.theme === "dark";
    apply(dark ? "light" : "dark");
  });

  var first = stocks[0];
  first.parentNode.insertBefore(btn, first);
});
