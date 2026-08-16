// Wikipedia's own "go" search: redirects straight to the article on an
// exact/near title match, falls back to full-text search results
// otherwise -- safer than guessing a direct /wiki/<Title> URL, since author
// names (stored "Last, First" -- see book.author) and book titles won't
// always match Wikipedia's own title casing/punctuation.
export function wikipediaSearchUrl(query) {
  return `https://en.wikipedia.org/w/index.php?title=Special:Search&search=${encodeURIComponent(query)}&fulltext=1&ns0=1`
}

export function googleSearchUrl(query) {
  return `https://www.google.com/search?q=${encodeURIComponent(query)}`
}
