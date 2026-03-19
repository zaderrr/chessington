import ErrorBoundary from "./ErrorBoundary"
const moveBoard = () => {
    return (
        <ErrorBoundary>
            <div className="w-1/4">
                <div>
                    <p>Header of the move board? </p>
                </div>
            </div>
        </ErrorBoundary>
    )
}

export default moveBoard
