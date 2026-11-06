// 成果網站ê選單佮 footer：囥佇 <gradio-app> 頭前佮後壁，
// 按呢 Gradio 家己ê <main> 就干焦包主內容。
// window.SA_KHING 由 common/utils.py 提供（header、footer ê HTML）。
(function () {
  'use strict';

  function tshiat(toggle, menu, show) {
    menu.classList.toggle('show', show);
    toggle.classList.toggle('show', show);
    toggle.setAttribute('aria-expanded', show ? 'true' : 'false');
  }

  function kuainnDropdown(sauLiau) {
    const menus = document.querySelectorAll('.sa-khing .dropdown-menu.show');
    for (const menu of menus) {
      const dropdown = menu.parentElement;
      if (sauLiau && dropdown.contains(sauLiau)) {
        continue;
      }
      tshiat(dropdown.querySelector('[data-bs-toggle="dropdown"]'), menu, false);
    }
  }

  // 取代 Bootstrap JS ê collapse 佮 dropdown
  document.addEventListener('click', function (event) {
    kuainnDropdown(event.target);
    const toggle = event.target.closest('.sa-khing [data-bs-toggle]');
    if (!toggle) {
      return;
    }
    if (toggle.dataset.bsToggle === 'dropdown') {
      const menu = toggle.parentElement.querySelector('.dropdown-menu');
      tshiat(toggle, menu, !menu.classList.contains('show'));
    } else if (toggle.dataset.bsToggle === 'collapse') {
      const target = document.querySelector(toggle.dataset.bsTarget);
      tshiat(toggle, target, !target.classList.contains('show'));
    }
  });

  document.addEventListener('keydown', function (event) {
    if (event.key !== 'Escape') {
      return;
    }
    const menu = document.querySelector('.sa-khing .dropdown-menu.show');
    if (menu) {
      kuainnDropdown(null);
      menu.parentElement.querySelector('[data-bs-toggle="dropdown"]').focus();
    }
  });

  // 瀏覽人次：Gradio ê demo.load 提著數字了後呼叫
  window.saKhingGaPv = function (pv) {
    const li = document.getElementById('ga-pv');
    if (li && pv !== null && pv !== undefined) {
      li.querySelector('.sa-ga-pv').textContent = pv;
      li.hidden = false;
    }
  };

  // 逐个元素包一層 .sa-khing 囥入去；sticky-top 愛是 <body> ê囝才黏會牢
  function tsahJip(html, sikTsing) {
    const template = document.createElement('template');
    template.innerHTML = html;
    for (const element of Array.from(template.content.children)) {
      const pau = document.createElement('div');
      pau.className = 'sa-khing';
      if (element.classList.contains('sticky-top')) {
        pau.classList.add('sa-sticky-top');
      }
      pau.append(element);
      sikTsing(pau);
    }
  }

  let tsahLiau = false;

  function tsahKhing() {
    const app = document.querySelector('gradio-app');
    if (tsahLiau || !app || !window.SA_KHING) {
      return;
    }
    tsahLiau = true;
    tsahJip(window.SA_KHING.header, function (pau) {
      app.before(pau);
    });
    const footer = window.SA_KHING.footer.replace(
      '{{ ni }}', new Date().getFullYear());
    tsahJip(footer, function (pau) {
      app.after(pau);
    });
  }

  // Gradio 是畫面載入了才插 head ê script，DOMContentLoaded 可能已經過矣
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', tsahKhing);
  } else {
    tsahKhing();
  }
})();
