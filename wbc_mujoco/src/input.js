const INTERACTIVE_TARGETS = [
  "button",
  "input",
  "select",
  "textarea",
  "a",
  "summary",
  '[contenteditable]:not([contenteditable="false"])',
].join(",");

export function shouldIgnoreGlobalShortcut(event) {
  if (
    !event
    || event.defaultPrevented
    || event.repeat
    || event.isComposing
    || event.ctrlKey
    || event.metaKey
    || event.altKey
  ) {
    return true;
  }
  return Boolean(event.target?.closest?.(INTERACTIVE_TARGETS));
}
