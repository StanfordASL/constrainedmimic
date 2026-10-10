document.addEventListener('DOMContentLoaded', function () {
  // Light/dark toggle. With no saved choice, the page follows the OS setting.
  var root = document.documentElement;
  var toggle = document.getElementById('theme-toggle');
  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)');

  function isDark() {
    var theme = root.getAttribute('data-theme');
    return theme ? theme === 'dark' : prefersDark.matches;
  }

  toggle.addEventListener('click', function () {
    var next = isDark() ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
  });

  function play(video) {
    var p = video.play();
    if (p) p.catch(function () {});
  }

  function restart(videos) {
    videos.forEach(function (v) { v.currentTime = 0; });
    videos.forEach(play);
  }

  // Videos only play while on screen. Videos inside a [data-sync] group restart
  // together when the group scrolls into view, so side-by-side comparisons line up.
  var groups = document.querySelectorAll('[data-sync]');
  var grouped = new Set();
  var groupObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      var videos = Array.prototype.slice.call(entry.target.querySelectorAll('video'));
      if (entry.isIntersecting) restart(videos);
      else videos.forEach(function (v) { v.pause(); });
    });
  }, { threshold: 0.25 });

  groups.forEach(function (group) {
    group.querySelectorAll('video').forEach(function (v) { grouped.add(v); });
    groupObserver.observe(group);
    var button = group.querySelector('.replay');
    if (button) {
      button.addEventListener('click', function () {
        restart(Array.prototype.slice.call(group.querySelectorAll('video')));
      });
    }
  });

  var videoObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) play(entry.target);
      else entry.target.pause();
    });
  }, { threshold: 0.25 });

  document.querySelectorAll('video').forEach(function (v) {
    if (!grouped.has(v)) videoObserver.observe(v);
  });

  // Copy buttons on code blocks.
  document.querySelectorAll('.copy').forEach(function (button) {
    button.addEventListener('click', function () {
      var text = button.parentElement.querySelector('code').innerText;
      navigator.clipboard.writeText(text).then(function () {
        button.textContent = 'Copied';
        setTimeout(function () { button.textContent = 'Copy'; }, 1500);
      });
    });
  });
});
