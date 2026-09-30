// Themed dropdown lists. Browsers draw <select> and <datalist> popups themselves (plain
// white menus that can't be styled), so the provider picker and the model suggestions
// use this instead: a sticker-style list that lines up under (or above) its field.

let current = null; // the open dropdown, if any

export class Dropdown {
  /**
   * anchor: the field the list belongs to (a button, or a text input that filters the list).
   * onPick(value) is called when an option is chosen.
   */
  constructor(anchor, { onPick, label = "Options" } = {}) {
    this.anchor = anchor;
    this.onPick = onPick || (() => {});
    this.items = [];
    this.shown = [];
    this.value = null;
    this.active = -1;
    this.list = document.createElement("ul");
    this.list.className = "dropdown";
    this.list.id = `${anchor.id}-list`;
    this.list.hidden = true;
    this.list.setAttribute("role", "listbox");
    this.list.setAttribute("aria-label", label);
    document.body.append(this.list);
    anchor.setAttribute("aria-controls", this.list.id);
    anchor.setAttribute("aria-expanded", "false");

    this.list.addEventListener("pointerdown", (e) => e.preventDefault()); // keep focus on the field
    this.list.addEventListener("click", (e) => {
      const li = e.target.closest("li[data-i]");
      if (li) this.pick(Number(li.dataset.i));
    });
    this.onOutside = (e) => { if (!this.list.contains(e.target) && !this.anchor.parentElement.contains(e.target)) this.close(); };
    this.onMove = () => this.position();
  }

  get isOpen() { return !this.list.hidden; }

  /** items: [{value, label?, hint?}] */
  setItems(items) {
    this.items = items;
    if (this.isOpen && !this.render(this.query)) this.close();
  }

  render(query = "") {
    this.query = query;
    // A text field's own text is its value; nothing is pre-selected unless it matches exactly,
    // so Enter keeps a custom name instead of jumping to a suggestion.
    const typing = this.anchor.tagName === "INPUT";
    const current = typing ? this.anchor.value.trim() || this.anchor.placeholder : this.value; // empty = the default
    const q = query.trim().toLowerCase();
    const exact = this.items.some((it) => it.value.toLowerCase() === q);
    this.shown = !q || exact ? this.items : this.items.filter((it) => `${it.value} ${it.label || ""}`.toLowerCase().includes(q));
    this.list.innerHTML = "";
    this.shown.forEach((it, i) => {
      const li = document.createElement("li");
      li.id = `${this.list.id}-${i}`;
      li.dataset.i = String(i);
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(it.value === current));
      const name = document.createElement("span");
      name.className = "dd-name";
      name.textContent = it.label || it.value;
      li.append(name);
      if (it.hint) {
        const hint = document.createElement("span");
        hint.className = "dd-hint";
        hint.textContent = it.hint;
        li.append(hint);
      }
      this.list.append(li);
    });
    const sel = this.shown.findIndex((it) => it.value === current);
    this.highlight(sel >= 0 || typing ? sel : 0);
    return this.shown.length > 0;
  }

  open(query = "") {
    if (!this.render(query)) { this.close(); return; }
    if (current && current !== this) current.close();
    current = this;
    if (!this.isOpen) {
      this.list.hidden = false;
      this.anchor.setAttribute("aria-expanded", "true");
      document.addEventListener("pointerdown", this.onOutside, true);
      window.addEventListener("resize", this.onMove);
      window.addEventListener("scroll", this.onMove, true);
      window.visualViewport?.addEventListener("resize", this.onMove);
    }
    this.position();
    this.list.children[this.active]?.scrollIntoView({ block: "nearest" });
  }

  close() {
    if (!this.isOpen) return;
    this.list.hidden = true;
    this.anchor.setAttribute("aria-expanded", "false");
    this.anchor.removeAttribute("aria-activedescendant");
    document.removeEventListener("pointerdown", this.onOutside, true);
    window.removeEventListener("resize", this.onMove);
    window.removeEventListener("scroll", this.onMove, true);
    window.visualViewport?.removeEventListener("resize", this.onMove);
    if (current === this) current = null;
  }

  toggle() { if (this.isOpen) this.close(); else this.open(); }

  highlight(i) {
    this.active = i;
    [...this.list.children].forEach((li, k) => li.classList.toggle("active", k === i));
    const li = this.list.children[i];
    if (li) {
      this.anchor.setAttribute("aria-activedescendant", li.id);
      li.scrollIntoView({ block: "nearest" });
    } else this.anchor.removeAttribute("aria-activedescendant");
  }

  pick(i) {
    const it = this.shown[i];
    if (!it) return;
    this.value = it.value;
    this.close();
    this.onPick(it.value);
  }

  /** Arrow keys, Enter and Escape; returns true when the key was used. */
  key(e) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!this.isOpen) this.open(this.anchor.tagName === "INPUT" ? this.anchor.value : "");
      else {
        const n = this.shown.length;
        this.highlight(this.active < 0 ? (e.key === "ArrowDown" ? 0 : n - 1) : (this.active + (e.key === "ArrowDown" ? 1 : -1) + n) % n);
      }
    } else if (e.key === "Enter" && this.isOpen && this.active >= 0) {
      this.pick(this.active);
    } else if (e.key === "Enter" && this.isOpen) {
      this.close(); // keep the typed text; the field's own change event saves it
      return false;
    } else if (e.key === "Escape" && this.isOpen) {
      this.close();
    } else return false;
    e.preventDefault();
    return true;
  }

  /** Line the list up with the field: below it, or above when there's more room there. */
  position() {
    const r = this.anchor.getBoundingClientRect();
    const vv = window.visualViewport;
    const top = vv ? vv.offsetTop : 0;
    const bottom = vv ? vv.offsetTop + vv.height : window.innerHeight;
    const below = bottom - r.bottom - 14;
    const above = r.top - top - 14;
    const want = Math.min(300, this.list.scrollHeight + 6); // the list's full height, capped
    const up = want > below && above > below;
    const s = this.list.style;
    s.left = `${Math.round(r.left)}px`;
    s.width = `${Math.round(r.width)}px`;
    s.maxHeight = `${Math.round(Math.max(120, Math.min(300, up ? above : below)))}px`;
    s.top = up ? "" : `${Math.round(r.bottom + 8)}px`;
    s.bottom = up ? `${Math.round(window.innerHeight - r.top + 8)}px` : "";
    this.list.classList.toggle("up", up);
  }
}
