import { isDark } from './domainColors'

// Colours for the Origins charts. Eight categorical hues at most, assigned by
// FAMILY in a fixed order (never by rank, so a genre filter never repaints a
// family): the six largest families get slots 1-6 of the validated default
// palette, every smaller language family shares slot 7 (told apart by labels,
// tooltips and the table), and the three non-languages (Mixed, Obscure,
// Imitative) are neutral grey. Validated 2026-11-07 with the dataviz
// validate_palette.js against this app's surfaces (#fff / #12161b): all
// adjacent checks pass in both modes; light mode's aqua, yellow and magenta
// sit under 3:1, so every chart ships direct labels and a table view.
const SLOTS = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9'],
}
const NEUTRAL = { light: '#a3a29c', dark: '#6b6a64' }

export const MAJOR = ['Latin', 'Old & Middle English', 'Greek', 'French', 'Dutch, German & Frankish', 'Norse & Scandinavian']
export const MINOR = [
  'Celtic', 'Italian', 'Spanish & Portuguese', 'Slavic', 'Hebrew & Semitic', 'Arabic, Persian & Turkic',
  'Indian languages', 'East Asian', 'Other',
]
export const NON_LANGUAGE = ['Mixed', 'Obscure origin', 'Imitative']
// Ring / treemap order: fixed, so neighbours (and the validated adjacent
// pairs) never change with the filter.
export const FAMILY_ORDER = [...MAJOR, ...MINOR, ...NON_LANGUAGE]

export function familyColor(family) {
  const mode = isDark() ? 'dark' : 'light'
  const i = MAJOR.indexOf(family)
  if (i >= 0) return SLOTS[mode][i]
  if (NON_LANGUAGE.includes(family)) return NEUTRAL[mode]
  return SLOTS[mode][6]
}

// The outer ring's route shading: the family hue, mixed toward the surface.
// Full = straight into English (or inherited); medium = came via another
// language; pale = formed in English from the family's words.
export const ROUTE_MIX = { inherited: 0, direct: 0, via: 30, formed: 55 }

export function routeKind(route) {
  return route.startsWith('via:') ? 'via' : route
}

export function routeColor(family, route) {
  const mix = ROUTE_MIX[routeKind(route)] ?? 0
  const base = familyColor(family)
  return mix ? `color-mix(in oklch, ${base}, var(--bg) ${mix}%)` : base
}

export function routeLabel(route) {
  if (route === 'inherited') return 'inherited from Old English'
  if (route === 'direct') return 'borrowed directly'
  if (route === 'formed') return 'formed in English'
  if (route.startsWith('via:')) return `via ${route.slice(4)}`
  return route
}
