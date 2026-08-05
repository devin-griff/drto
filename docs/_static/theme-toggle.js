/* A two-state light/dark button: the moon shows in light mode, the
   sun in dark mode (CSS on html[data-theme] picks the icon), and one
   click toggles. Writes the same localStorage keys and <html>
   attributes the theme itself uses, so everything styled by
   data-theme follows. */
document.addEventListener("DOMContentLoaded", function () {
  var stock = document.querySelector("button.theme-switch-button");
  if (!stock || !stock.parentNode) {
    return;
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
  btn.id = "drto-theme-toggle";
  btn.className = "btn btn-sm nav-link pst-navbar-icon";
  btn.innerHTML =
    '<i class="fa-solid fa-moon fa-lg"></i>' +
    '<i class="fa-solid fa-sun fa-lg"></i>';

  function tooltip() {
    var dark = document.documentElement.dataset.theme === "dark";
    btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
    btn.setAttribute("aria-label", btn.title);
  }

  btn.addEventListener("click", function () {
    var dark = document.documentElement.dataset.theme === "dark";
    apply(dark ? "light" : "dark");
    tooltip();
  });

  tooltip();
  stock.parentNode.insertBefore(btn, stock);
});
