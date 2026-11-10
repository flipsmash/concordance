import { Link, useNavigate } from 'react-router-dom'
import AuthorSelect from './AuthorSelect'
import BookSelect from './BookSelect'
import { VizPage, VizPageHeader } from './VizPage'
import VizThumb from './VizThumb'
import './Browse.css' // .author-select* for the pickers
import './Visualizations.css'

// The Visualizations hub: a gallery of cards, grouped by what they're about,
// each opening its own page. It used to embed every chart in one ~5,000px
// scroll (growth, six histograms, two force graphs, a word graph), mixed
// with plain links; the charts now live on their own pages.
function Card({ to, thumb, title, children, extra = null }) {
  return (
    <article className="viz-card">
      <Link to={to} className="viz-card-main">
        <VizThumb kind={thumb} />
        <h3 className="viz-card-title">{title}</h3>
        <p className="viz-card-text">{children}</p>
      </Link>
      {extra && <div className="viz-card-extra">{extra}</div>}
    </article>
  )
}

function Group({ title, children }) {
  return (
    <section className="viz-group">
      <h2 className="viz-group-title">{title}</h2>
      <div className="viz-grid">{children}</div>
    </section>
  )
}

function Visualizations() {
  const navigate = useNavigate()
  return (
    <VizPage wide>
      <VizPageHeader title="Visualizations" back={false}>
        Pictures of the collection: how its books and authors relate, where its words come from, and how it grew.
      </VizPageHeader>

      <Group title="How books and authors relate">
        <Card to="/app/books/relatedness" thumb="network" title="Books by shared vocabulary"
          extra={
            <>
              <BookSelect placeholder="Find a book…"
                onPick={(book) => navigate(`/app/authors/${encodeURIComponent(book.author || '')}/${book.id}/relatedness`)} />
              <Link to="/app/books/relatedness?tab=everything" className="viz-card-link">Map every book →</Link>
            </>
          }>
          Pick a book to see the books that share the most rare words with it, or see the top 200 at once.
        </Card>
        <Card to="/app/authors/relatedness" thumb="network" title="Authors by shared vocabulary"
          extra={
            <>
              <AuthorSelect value={null}
                onChange={(author) => author && navigate(`/app/authors/${encodeURIComponent(author)}/relatedness`)} />
              <Link to="/app/authors/relatedness?tab=everything" className="viz-card-link">Map every author →</Link>
            </>
          }>
          Pick an author to see who writes with the most similar rare vocabulary, or see the top 200 at once.
        </Card>
        <Card to="/app/books/relatedness?tab=everything" thumb="scatter" title="Every book on one map">
          The whole collection in one zoomable map, grouped and named by the words that set each group apart.
        </Card>
        <Card to="/app/books/relatedness?tab=force" thumb="network" title="Books as a force map"
          extra={<Link to="/app/authors/relatedness?tab=force" className="viz-card-link">Authors as a force map →</Link>}>
          Every book pulled toward the books it shares the most rare words with, into a zoomable map of
          neighbourhoods.
        </Card>
        <Card to="/app/visualizations/genres" thumb="network" title="Genres by shared vocabulary">
          Which genres share the most rare words &mdash; from niche genres to genre fiction to the classics.
        </Card>
        <Card to="/app/visualizations/domain-map" thumb="field" title="Works by subject field">
          Works and authors placed by how their words spread across science, law, the arts and more.
        </Card>
      </Group>

      <Group title="Words and meaning">
        <Card to="/app/visualizations/origins" thumb="sunburst" title="Where the words come from">
          Latin, Old English, Greek, French, Norse&hellip; each word&rsquo;s source language and the route it took
          into English.
        </Card>
        <Card to="/app/visualizations/category-tree" thumb="tree" title="Categories by meaning">
          Every subject category, clustered by what its words mean, with its words listed alongside.
        </Card>
        <Card to="/app/visualizations/word-graph" thumb="network" title="Words by meaning">
          Search a word to see its nearest neighbours by definition or by spelling.
        </Card>
        <Card to="/app/visualizations/words-map" thumb="scatter" title="Every word by meaning">
          The whole vocabulary on one zoomable map, words gathered into neighbourhoods by what they mean.
        </Card>
        <Card to="/app/visualizations/popularity" thumb="line" title="Popularity over time">
          How often words appeared in English books, decade by decade since 1800 &mdash; compare up to four.
        </Card>
      </Group>

      <Group title="The collection">
        <Card to="/app/visualizations/growth" thumb="bars" title="Corpus growth">
          Books, authors and words added week by week, or as a running total.
        </Card>
        <Card to="/app/visualizations/distributions" thumb="bars" title="Uniqueness, fame and difficulty">
          How many words each book or author has to itself, how famous they are, and how hard their words are.
        </Card>
      </Group>
    </VizPage>
  )
}

export default Visualizations
